#!/bin/bash
#SBATCH --job-name=bbrec
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/bumblebee/reccheck-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/bumblebee/reccheck-%j.out
#
#   sbatch --export=ALL,STEP=b0_default recorder_check.sbatch
#
# Recorder check in this image with a trivial program: (1) loader accepts the library, (2) the
# payload's allocator is intercepted, (3) host-side pmsample reads the table.
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/bumblebee
AR=$C/attrib_resident
SIF=$B/sif/bumblebee-$STEP.sif
OUT=$B/reccheck/$STEP

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT" || exit 1

echo "=== 1. does this image's loader accept the recorder"
apptainer exec --bind "$AR":/ar "$SIF" \
    bash -c 'LD_PRELOAD=/ar/pmrec.so /bin/echo loader-accepted-it' \
    || { echo "FAILED: the image refuses /ar/pmrec.so; rebuild it against an older glibc"; exit 2; }

echo "=== glibc the image has, and what the recorder requires"
apptainer exec "$SIF" bash -c 'ldd --version | head -1'
apptainer exec --bind "$AR":/ar "$SIF" \
    bash -c 'objdump -T /ar/pmrec.so | grep -o "GLIBC_[0-9.]*" | sort -u | tr "\n" " "; echo'

echo
echo "=== 2. does it record anything in THIS image, from a C++ program using operator new"
cat > "$OUT/probe.cc" <<'EOF'
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
int main() {
    // Four allocations well above the campaign floor, through the two paths a C++ program uses.
    char* a = new char[64u << 20];
    char* b = static_cast<char*>(malloc(48u << 20));
    for (size_t i = 0; i < (64u << 20); i += 4096) a[i] = 1;
    for (size_t i = 0; i < (48u << 20); i += 4096) b[i] = 1;
    fprintf(stderr, "PROBE_READY pid=%d\n", (int)getpid());
    fflush(stderr);
    sleep(20);
    delete[] a; free(b);
    return 0;
}
EOF
apptainer exec --bind "$OUT":"$OUT" "$SIF" \
    bash -c "g++ -O0 -g -o $OUT/probe $OUT/probe.cc" \
    || { echo "FAILED: could not compile the probe inside the image"; exit 3; }

apptainer exec --bind "$AR":/ar --bind "$OUT":"$OUT" "$SIF" bash -c \
    "export LD_PRELOAD=/ar/pmrec.so SNNI_PM_TABLE=$OUT/pmtable_%d SNNI_PM_MIN=${SNNI_PM_MIN:?campaign.env must supply the recorder floor}; exec $OUT/probe" \
    > "$OUT/probe.out" 2> "$OUT/probe.err" &
APP=$!

PID=""
for _ in $(seq 1 200); do
    PID=$(pgrep -u "$(id -u)" -f "$OUT/probe" | while read -r p; do
              case "$(readlink "/proc/$p/exe" 2>/dev/null)" in */probe) echo "$p";; esac
          done | head -1)
    [ -n "$PID" ] && [ -s "$OUT/pmtable_$PID" ] && break
    sleep 0.1
done
if [ -z "$PID" ]; then
    echo "FAILED: the probe never appeared"; kill "$APP" 2>/dev/null; exit 4
fi
echo "probe pid=$PID table=$(ls -la "$OUT"/pmtable_* 2>/dev/null)"

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
# 112 MiB was allocated and touched; a resident total far below that means the recorder saw
# nothing, which looks exactly like a correct (but empty) measurement.
grep -h '^# live_allocations' "$OUT/resident.txt"
TOT=$(awk '!/^#/{s+=$1} END{print s+0}' "$OUT/resident.txt" 2>/dev/null)
echo "recorded RESIDENT bytes across all sites = $TOT  (112 MiB = 117440512 was touched)"
[ "$TOT" -ge 100000000 ] \
    || { echo "FAILED: the recorder loaded but recorded almost nothing, which is the failure shape"
         echo "        that reads exactly like a correct measurement. Do not measure on this."; exit 6; }
echo "RECCHECK_DONE"
