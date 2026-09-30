#!/bin/bash
#SBATCH --job-name=moai-gpu-linkrun
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/diag/linkrun-%A_%a.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/diag/linkrun-%A_%a.err
#
# Diagnostic. Both link-mode builds, three runs each (same-binary fragmentation band +-1.5 to
# 2.5 GiB). Reports GPUPEAK (driver-maintained, read by make_steps.py) and the 100 ms nvidia-smi poll.
set -uo pipefail

BASE=/scratch/tmp/j_wred02/snni_campaign/moai-gpu
DIAG=$BASE/diag
SRC=$DIAG/src-pristine
SIF=/scratch/tmp/j_wred02/moai_gpu/moai-gpu-s0_default.sif

case $((SLURM_ARRAY_TASK_ID / 3)) in
    0) VARIANT=build-static ;;
    1) VARIANT=build-shared ;;
    *) echo "bad array index" >&2; exit 2 ;;
esac
REP=$((SLURM_ARRAY_TASK_ID % 3))
RES=$DIAG/runs/$VARIANT/run$REP
rm -rf "$RES"; mkdir -p "$RES" || exit 2

BIN=$SRC/$VARIANT/test
[ -x "$BIN" ] || { echo "no binary at $BIN" >&2; exit 2; }

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

{
    echo "variant=$VARIANT"
    echo "replicate=$REP"
    echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    echo "host=$(hostname)"
    echo "binary=$BIN"
    echo "binary_md5=$(md5sum "$BIN" | awk '{print $1}')"
    echo "image=$SIF"
    echo "image_digest=$(sha256sum "$SIF" | awk '{print $1}')"
    echo "omp_num_threads=8"
    echo "workload=BERT-base 1 encoder layer, batch 256, seq 128"
} > "$RES/RUN_META"

nvidia-smi -L >> "$RES/RUN_META" 2>&1

# Address the GPU by UUID, not the logical/physical index pair that `-i` conflates.
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')
echo "gpu_uuid=$GPU_UUID" >> "$RES/RUN_META"
[ -n "$GPU_UUID" ] || { echo "ERROR: no GPU visible to this job" >&2; exit 2; }

# 100 ms device poller, identical for both variants.
VPOLL="$RES/vram_poll.log"; : > "$VPOLL"
(
  while true; do
    ts=$(date +%s%3N)
    used=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    echo "$ts ${used:-NA}" >> "$VPOLL"
    sleep 0.1
  done
) & VPOLLER=$!
trap 'kill $VPOLLER 2>/dev/null' EXIT INT TERM

T0=$(date +%s)
# Set for both builds (the static one ignores it) so the two command lines stay identical.
apptainer exec --nv --bind /scratch:/scratch "$SIF" bash -lc "
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\${LD_LIBRARY_PATH:-}
    export OMP_NUM_THREADS=8
    cd $SRC/$VARIANT
    exec ./test
" > "$RES/stdout.log" 2> "$RES/stderr.log"
RC=$?
T1=$(date +%s)

kill $VPOLLER 2>/dev/null
echo "$RC" > "$RES/exit_status.txt"
echo $((T1 - T0)) > "$RES/wall_seconds.txt"

grep -E "^MEM\|" "$RES/stdout.log" > "$RES/markers.log" 2>/dev/null
grep -E "^(GPUPEAK|HOSTPEAK)\|" "$RES/stdout.log" > "$RES/all_markers.txt" 2>/dev/null
grep -E "Modulus chain index" "$RES/stdout.log" > "$RES/levels.txt" 2>/dev/null

# Guards: the first attempt ran the wrong program (RMSNorm unit test, 14s, exit 0, plausible peak)
# and nothing failed, so each assertion here names a quantity that can't be right by accident.
NMARK=$(grep -c '^MEM|' "$RES/markers.log" 2>/dev/null || echo 0)
NPEAK=$(grep -c '^GPUPEAK|' "$RES/all_markers.txt" 2>/dev/null || echo 0)
WALL=$(cat "$RES/wall_seconds.txt")

if [ "$NMARK" -lt 16 ]; then
    echo "FAILED: $NMARK of 16 phase markers. The instrumented single-layer path did not run." >&2
    RC=6
fi
if [ "$NPEAK" -lt 1 ]; then
    echo "FAILED: no GPUPEAK line. Without it there is no device peak for make_steps.py to read." >&2
    RC=6
fi
# Workload is ~2700s on this card; under ten minutes is a different program regardless of output.
if [ "$WALL" -lt 600 ]; then
    echo "FAILED: wall time ${WALL}s is far below the ~2700s this workload takes." >&2
    RC=6
fi
if [ "$RC" -eq 6 ]; then
    echo "=== stdout head, to see what actually ran ===" >&2
    head -5 "$RES/stdout.log" >&2
fi
echo "$RC" > "$RES/exit_status.txt"

{
    echo "variant=$VARIANT replicate=$REP exit=$RC wall_s=$((T1 - T0))"
    echo "-- exact, driver-maintained high-water --"
    cat "$RES/all_markers.txt" 2>/dev/null
    echo "-- 100 ms poll --"
    awk '{if($2!="NA" && $2+0>m)m=$2+0} END{printf "poll_vram_peak_MiB=%d (%.2f GiB)\n", m, m/1024}' "$VPOLL"
} | tee "$RES/PEAK.txt"

echo "=== markers ==="; cat "$RES/markers.log"
echo "=== levels ==="; cat "$RES/levels.txt"
exit $RC
