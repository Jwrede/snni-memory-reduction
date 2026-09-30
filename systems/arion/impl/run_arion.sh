#!/bin/bash
# ARION, inside the container. One process, no parties (pure FHE is non-interactive). The binary is
# a menu reading its task number from stdin, so a WORKLOAD marker is checked afterwards: a wrong
# menu number still runs, exits 0 and reports a plausible peak.
set -uo pipefail

RESULTS="${ARION_RESULTS:?need ARION_RESULTS}"
MENU="${ARION_MENU:-12}"          # 12 = BERT-base, multi-threaded, logN=16, rows=128, batch=256
BIN=/root/arion/arion

mkdir -p "$RESULTS"
rm -f "$RESULTS/arion.pid" "$RESULTS/gate.txt"
md5sum "$BIN" | tee "$RESULTS/binary_md5.txt"

cd /root/arion || exit 2
# Per-layer output lands under Output/<model>/ (bound to the results dir) so the observable survives.
mkdir -p /root/arion/Output

echo "=== arion menu=$MENU threads=${ARION_THREADS:-<default 64>} layers=${ARION_N_LAYERS:-<all>}"
T0=$(date +%s)

# GODEBUG=gctrace=1 prints live-heap-after-GC per cycle: the closest thing to attribution on a Go
# system, published as evidence beside the VmHWM peak, never as the peak.
printf '%s\n0\n' "$MENU" | env GODEBUG=gctrace=1 "$BIN" \
    > "$RESULTS/stdout.log" 2> "$RESULTS/markers.txt" &
PID=$!
echo "$PID" > "$RESULTS/arion.pid"
echo "arion pid=$PID"

RC=0; wait "$PID" || RC=$?
echo "=== arion rc=$RC wall=$(( $(date +%s) - T0 ))s"
echo "$RC" > "$RESULTS/program_exit_code.txt"

# --- the workload marker: did the menu actually start the run we asked for -------------------
grep -h '^WORKLOAD|'   "$RESULTS/markers.txt" > "$RESULTS/workload.txt"   2>/dev/null
grep -h '^SNNI_PARAM|' "$RESULTS/markers.txt" > "$RESULTS/snni_params.txt" 2>/dev/null
if [ ! -s "$RESULTS/workload.txt" ]; then
    echo "FAILED: no WORKLOAD| marker; the menu did not enter the measured workload" >&2
    [ "$RC" = 0 ] && RC=5
fi

# --- the gate observable ----------------------------------------------------------------------
# ARION writes each layer's output itself (Output/<model>/layer_<i>_output.csv), so the gate
# observable exists upstream unpatched; flattened to one value per line.
python3 - "$RESULTS" <<'PYEOF'
import glob, os, sys
res = sys.argv[1]
files = sorted(glob.glob("/root/arion/Output/**/layer_*_output.csv", recursive=True))
n = 0
nz = 0
with open(os.path.join(res, "gate.txt"), "w") as out:
    for f in files:
        for line in open(f):
            for tok in line.replace(",", " ").split():
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
cat >> "$RESULTS/markers.txt" < /dev/null

NG=$(wc -l < "$RESULTS/gate.txt" 2>/dev/null || echo 0)
echo "gate values: $NG"
if [ "$NG" -lt 1000 ]; then
    echo "FAILED: gate observable missing or short ($NG values); the run produced no decrypted output" >&2
    [ "$RC" = 0 ] && RC=5
fi
# A degenerate (all-zero) observable is not an observable; SHARK published one before this check existed.
if ! awk 'BEGIN{nz=0} {if ($1+0 != 0) nz++} END{exit !(nz>0)}' "$RESULTS/gate.txt"; then
    echo "FAILED: every gate value is zero" >&2
    [ "$RC" = 0 ] && RC=6
fi

# The Go collector's own view of the working set, for the MANIFEST's evidence section.
grep -oE '[0-9]+->[0-9]+->[0-9]+ MB' "$RESULTS/markers.txt" 2>/dev/null \
    | awk -F'->' '{split($3,a," "); if (a[1]+0>m) m=a[1]+0} END{printf "GCTRACE|max_live_after_gc_mb=%d\n", m+0}' \
    | tee -a "$RESULTS/workload.txt"

exit "$RC"
