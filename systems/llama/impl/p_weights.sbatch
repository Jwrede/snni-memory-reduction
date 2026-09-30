#!/bin/bash
#SBATCH --job-name=llweights
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/llweights-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/llweights-%j.err

# Source introspection (o0 image): lazy-weight flag and size floor uses, FC layer init/shrink/
# re-materialise in layers.h. Nothing measured or published.
set -uo pipefail

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

SIF="${SIF:-/scratch/tmp/j_wred02/snni_campaign/llama/sif/llama-online-o0_default.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

apptainer exec "$SIF" bash -c '
R=/opt/EzPC/GPU-MPC/ext/sytorch

echo "=== every use of the lazy flag and the size floor ==="
grep -n "LAZY_W_MIN_ELEMS\|g_lazyWeights\|g_weightSeedCtr" $R/include/sytorch/layers/layers.h

echo
echo "=== the FC layer: init, shrink and re-materialise, verbatim ==="
awk "/class FC/,/^};/" $R/include/sytorch/layers/layers.h | head -120
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
