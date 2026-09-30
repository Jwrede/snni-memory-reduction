#!/bin/bash
#SBATCH --job-name=pumarec
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/puma/reccheck-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/puma/reccheck-%j.out
#
#   sbatch --export=ALL,STEP=s0_default recorder_check.sbatch
#
# Recorder check in this image with a trivial program: (1) loader accepts the library, (2) a Python
# payload's allocations are recorded, (3) host-side pmsample reads the table, (4) allocation entry
# points imported by libspu.
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/puma
AR=$C/attrib_resident
SIF=$B/sif/puma-$STEP.sif
OUT=$B/reccheck/$STEP

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT" || exit 1

echo "=== 1. does this image's loader accept the recorder"
apptainer exec --bind "$AR":/ar "$SIF" \
    bash -c 'LD_PRELOAD=/ar/pmrec.so /bin/echo loader-accepted-it' \
    || { echo "FAILED: the image refuses /ar/pmrec.so; rebuild it against an older glibc"; exit 2; }
apptainer exec "$SIF" bash -c 'ldd --version | head -1'

echo
echo "=== 4. which allocation entry points does the measured stack import"
# Done before the probe: an unhooked allocation path could let the probe pass while blind.
apptainer exec "$SIF" bash -c '
for so in $(python3 -c "
import spu, os, glob
d = os.path.dirname(spu.__file__)
print(\"\n\".join(glob.glob(os.path.join(d, \"**\", \"*.so\"), recursive=True)))
" | head -20); do
    echo "--- $so"
    nm -D "$so" 2>/dev/null | grep -E " U (malloc|calloc|realloc|aligned_alloc|posix_memalign|memalign|valloc|_Znwm|_Znam)(@.*)?$" | sort -u
done' | tee "$OUT/alloc_imports.txt"

if grep -qE " U (aligned_alloc|memalign|valloc)$" "$OUT/alloc_imports.txt"; then
    echo "NOTE: this stack imports an aligned allocator. pmrec has hooked aligned_alloc since"
    echo "      2026-07-30; any table taken with an earlier build of it is invalid for this system."
fi

echo
echo "=== 2. does it record what THIS image's Python actually allocates"
cat > "$OUT/probe.py" <<'EOF'
import os, sys, time
import numpy as np
# 112 MiB in two arrays, both well above the campaign floor, both TOUCHED so they are resident.
a = np.ones(64 << 20, dtype=np.uint8)
b = np.ones(48 << 20, dtype=np.uint8)
a[::4096] = 2
b[::4096] = 2
sys.stderr.write("PROBE_READY pid=%d\n" % os.getpid())
sys.stderr.flush()
time.sleep(20)
print(int(a.sum() > 0), int(b.sum() > 0))
EOF

apptainer exec --bind "$AR":/ar --bind "$OUT":"$OUT" "$SIF" bash -c \
    "export LD_PRELOAD=/ar/pmrec.so SNNI_PM_TABLE=$OUT/pmtable_%d SNNI_PM_MIN=${SNNI_PM_MIN:?campaign.env must supply the recorder floor}; exec python3 $OUT/probe.py" \
    > "$OUT/probe.out" 2> "$OUT/probe.err" &
APP=$!

# Wait for PROBE_READY, not the table file: the table exists before main runs (interposer init).
PID=""
for _ in $(seq 1 600); do
    PID=$(pgrep -u "$(id -u)" -f "$OUT/probe.py" | while read -r p; do
              case "$(readlink "/proc/$p/exe" 2>/dev/null)" in */python3*) echo "$p";; esac
          done | head -1)
    [ -n "$PID" ] && grep -q PROBE_READY "$OUT/probe.err" 2>/dev/null && break
    sleep 0.1
done
if [ -z "$PID" ]; then
    echo "FAILED: the probe never appeared"; cat "$OUT/probe.err"; kill "$APP" 2>/dev/null; exit 4
fi
echo "probe pid=$PID"; ls -la "$OUT"/pmtable_* 2>/dev/null

echo
echo "=== 3. can pmsample, running on the HOST, read that table"
: "${SNNI_PM_FULLSCAN:?campaign.env must supply SNNI_PM_FULLSCAN}"
export SNNI_PM_FULLSCAN
"$AR/pmsample" "$PID" "$OUT/pmtable_$PID" "$OUT/pm.maps" > "$OUT/resident.txt" 2>"$OUT/pmsample.err"
RC=$?
echo "pmsample rc=$RC"
head -20 "$OUT/resident.txt"
cat "$OUT/pmsample.err"
wait "$APP" 2>/dev/null

[ "$RC" = 0 ] || { echo "FAILED: pmsample could not read the table"; exit 5; }
grep -h '^# live_allocations' "$OUT/resident.txt"
TOT=$(awk '!/^#/{s+=$1} END{print s+0}' "$OUT/resident.txt" 2>/dev/null)
echo "recorded RESIDENT bytes across all sites = $TOT  (112 MiB = 117440512 was touched)"
[ "$TOT" -ge 100000000 ] \
    || { echo "FAILED: the recorder loaded but recorded almost nothing, which is the failure shape"
         echo "        that reads exactly like a correct measurement. Do not measure on this."; exit 6; }
echo "RECCHECK_DONE"
