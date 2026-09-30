#!/bin/bash
#SBATCH --job-name=moaicpu-flooratt
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/floor-att-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/floor-att-%j.out

# Runs the MOAI attention-instant micro-benchmark: FLOOR| lines (per-phase VmRSS/VmHWM) and
# FLOOR_SUMMARY|; no poller; SLURM MaxRSS printed as cross-check. T=1.
set -uo pipefail

B=/scratch/tmp/j_wred02/snni_campaign/moai-cpu
SIF="${SIF:-$B/sif/moai-cpu-p_floor_bench_att.sif}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
[ -r "$SIF" ] || { echo "FAILED: no floor sif at $SIF" >&2; exit 2; }

echo "=== node: $(hostname)  mem: $(free -g | awk '/Mem:/{print $2}') GB  $(date -u) ==="
# --pwd: the test binary reads its data files relative to /root/moai/build; the bench returns
# before the inference so it needs none, but this keeps startup identical to a normal run.
apptainer exec --pwd /root/moai/build \
    --env SNNI_FLOOR_ATT=1 --env OMP_NUM_THREADS=1 \
    "$SIF" /root/moai/build/test 2>&1

echo "=== done $(date -u) ==="
