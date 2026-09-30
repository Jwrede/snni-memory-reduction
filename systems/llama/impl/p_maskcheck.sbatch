#!/bin/bash
#SBATCH --job-name=llmask
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/llmask-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/llmask-%j.err

# Source introspection (o0 image): does initializeInferencePartyA modify the weights; FC layer
# initialisation. Nothing measured or published.
set -uo pipefail

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

SIF="${SIF:-/scratch/tmp/j_wred02/snni_campaign/llama/sif/llama-online-o0_default.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

apptainer exec "$SIF" bash -c '
R=/opt/EzPC/GPU-MPC/ext/sytorch
echo "=== does initializeInferencePartyA touch the weights ==="
grep -rn "initializeInferencePartyA" $R/include $R/ext/llama 2>/dev/null | head -6
F=$(grep -rln "void initializeInferencePartyA" $R/include $R/ext/llama 2>/dev/null | head -1)
echo "--- $F"
grep -n -A30 "void initializeInferencePartyA" "$F" 2>/dev/null | head -40
echo
echo "=== and what the FC layer does at that moment (initialize / getweights users) ==="
grep -rn "getweights()\|->initialize\|initializeWeights" $R/include/sytorch 2>/dev/null | head -12
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
