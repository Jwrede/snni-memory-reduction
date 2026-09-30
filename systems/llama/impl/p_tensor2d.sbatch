#!/bin/bash
#SBATCH --job-name=lltensor
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/lltensor-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/lltensor-%j.err

# Source introspection (o0 image): the Tensor2D class definition in tensor.h. Nothing measured or
# published.
set -uo pipefail

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

SIF="${SIF:-/scratch/tmp/j_wred02/snni_campaign/llama/sif/llama-online-o0_default.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

apptainer exec "$SIF" bash -c '
F=/opt/EzPC/GPU-MPC/ext/sytorch/include/sytorch/tensor.h
echo "=== where the real Tensor2D definition starts ==="
grep -n "class Tensor2D" $F
N=$(grep -n "class Tensor2D$" $F | tail -1 | cut -d: -f1)
[ -n "$N" ] || N=$(grep -n "class Tensor2D" $F | tail -1 | cut -d: -f1)
echo "--- from line $N"
sed -n "${N},$((N+95))p" $F
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
