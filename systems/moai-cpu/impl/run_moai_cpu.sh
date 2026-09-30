#!/bin/bash
# MOAI-CPU, inside the container. One process, no parties.
# The program reads/writes relative paths, so cwd is a writable dir holding symlinks to the image's
# read-only weight data (no writable overlay needed). The gate is upstream's own per-layer decrypted
# output (layer_<id>.txt); nothing was added to the computation to obtain the observable.
set -uo pipefail

RESULTS="${MOAI_RESULTS:?need MOAI_RESULTS}"
BIN=/root/moai/build/test
DATA=/root/moai/data
WORK="$RESULTS/work"

mkdir -p "$WORK"
rm -f "$RESULTS/moai.pid" "$RESULTS/gate.txt"
md5sum "$BIN" | tee "$RESULTS/binary_md5.txt"

# Symlink every top-level entry of the data directory into the writable working directory.
for d in "$DATA"/*; do
    ln -sfn "$d" "$WORK/$(basename "$d")"
done
ls "$WORK" | head -5

cd "$WORK" || exit 2
echo "=== moai-cpu layers=${MOAI_N_LAYERS:-<default 12>} threads=${OMP_NUM_THREADS:-<unset>} seed=${SNNI_SEED:-<unset>}"
T0=$(date +%s)

"$BIN" > "$RESULTS/stdout.log" 2> "$RESULTS/markers.txt" &
PID=$!
echo "$PID" > "$RESULTS/moai.pid"
echo "moai pid=$PID"

RC=0; wait "$PID" || RC=$?
echo "=== moai rc=$RC wall=$(( $(date +%s) - T0 ))s"
echo "$RC" > "$RESULTS/program_exit_code.txt"

# --- the markers, each checked separately ------------------------------------------------------
grep -h '^SEEDED|'     "$RESULTS/markers.txt" > "$RESULTS/seeded.txt"      2>/dev/null
grep -h '^WORKLOAD|'   "$RESULTS/markers.txt" > "$RESULTS/workload.txt"    2>/dev/null
grep -h '^SNNI_PARAM|' "$RESULTS/markers.txt" >> "$RESULTS/workload.txt"   2>/dev/null
grep -h '^MEM|'        "$RESULTS/markers.txt" > "$RESULTS/all_markers.txt" 2>/dev/null
if [ ! -s "$RESULTS/seeded.txt" ]; then
    echo "FAILED: no SEEDED| marker; SEAL's PRNG is unpinned in this run and no gate can compare" >&2
    [ "$RC" = 0 ] && RC=5
fi
if [ ! -s "$RESULTS/workload.txt" ]; then
    echo "FAILED: no WORKLOAD| marker; what ran is not recorded" >&2
    [ "$RC" = 0 ] && RC=5
fi

# --- the gate: upstream's per-layer decrypted output --------------------------------------------
python3 - "$WORK" "$RESULTS" <<'PYEOF'
import glob, os, re, sys
work, res = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(work, "layer_*.txt")),
               key=lambda f: int(re.search(r"layer_(\d+)\.txt$", f).group(1)))
n = nz = 0
with open(os.path.join(res, "gate.txt"), "w") as out:
    for f in files:
        for line in open(f, errors="replace"):
            # "<k>-th ciphertext: v, v, v, ..."
            body = line.split(":", 1)[1] if ":" in line else line
            for tok in body.replace(",", " ").split():
                try:
                    v = float(tok)
                except ValueError:
                    continue
                out.write("%.17g\n" % v)
                n += 1
                if v != 0.0:
                    nz += 1
sys.stderr.write("GATE|layer_outputs|files=%d|n=%d\n" % (len(files), n))
sys.stderr.write("GATE_NONZERO|%d/%d\n" % (nz, n))
if n and nz == 0:
    sys.stderr.write("GATE_DEGENERATE|every value is zero\n")
PYEOF

NG=$(wc -l < "$RESULTS/gate.txt" 2>/dev/null || echo 0)
echo "gate values: $NG"
if [ "$NG" -lt 1000 ]; then
    echo "FAILED: gate observable missing or short ($NG values); the run decrypted nothing" >&2
    [ "$RC" = 0 ] && RC=5
fi
# A gate of all zeros cannot fail: SHARK published one before this check existed.
if [ "$NG" -gt 0 ] && ! awk 'BEGIN{nz=0} {if ($1+0 != 0) nz++} END{exit !(nz>0)}' "$RESULTS/gate.txt"; then
    echo "FAILED: every gate value is zero" >&2
    [ "$RC" = 0 ] && RC=6
fi

# The per-layer files themselves are evidence and are kept; the symlink farm is not.
mkdir -p "$RESULTS/layer_outputs"
for f in "$WORK"/layer_*.txt; do [ -f "$f" ] && mv "$f" "$RESULTS/layer_outputs/"; done
find "$WORK" -maxdepth 1 -type l -delete

exit "$RC"
