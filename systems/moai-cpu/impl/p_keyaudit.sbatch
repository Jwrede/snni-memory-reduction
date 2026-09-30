#!/bin/bash
#SBATCH --job-name=moaicpu-keya
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/keyaudit-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/keyaudit-%j.out

# Key audit: probe image at every parameter's measured value (both create_galois_keys calls finish
# before batch-dependent work); cut short after the audit lines.
set -uo pipefail

B=/scratch/tmp/j_wred02/snni_campaign/moai-cpu
SIF="$B/sif/moai-cpu-p_keyaudit.sif"
[ -r "$SIF" ] || { echo "FAILED: no probe image at $SIF" >&2; exit 2; }

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

RES="$B/results/p_keyaudit"
RUNNER="$B/run_moai_cpu.sh"
[ -r "$RUNNER" ] || { echo "FAILED: no runner at $RUNNER" >&2; exit 2; }
rm -rf "$RES"; mkdir -p "$RES"

# Through the system's own runner, not a hand-written apptainer line: a direct invocation reads
# weight data by a relative path that fails silently outside the runner's working directory.
INNER="export MOAI_RESULTS=$RES MOAI_N_LAYERS=1 OMP_NUM_THREADS=16 SNNI_SEED=0x5EED"
INNER="$INNER; exec bash $RUNNER"

echo "=== running the probe image through run_moai_cpu.sh ==="
apptainer exec \
    --bind "$RES":"$RES" \
    --bind "$RUNNER":"$RUNNER" \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!
echo "apptainer pid=$APP_PID"

# Both markers.txt and stderr.log are watched, since which stream carries a message is the runner's
# choice, not something this job should assume.
for _ in $(seq 1 4200); do
    if grep -qs "KEYAUDIT|gal_keys_boot|" "$RES/markers.txt" "$RES/stderr.log"; then
        echo "audit complete after ${SECONDS}s; stopping the run"
        kill "$APP_PID" 2>/dev/null
        break
    fi
    kill -0 "$APP_PID" 2>/dev/null || { echo "the run ended on its own after ${SECONDS}s"; break; }
    sleep 5
done
wait "$APP_PID" 2>/dev/null

echo
echo "=== what the run reported ==="
grep -hE "KEYAUDIT\||POOL\||MEM\||SNNI_PARAM\||SEEDED\|" "$RES/markers.txt" "$RES/stderr.log" || {
    echo "FAILED: no audit lines at all; the probe instrumentation did not run" >&2
    tail -20 "$RES/markers.txt" "$RES/stderr.log" >&2
    exit 3
}
