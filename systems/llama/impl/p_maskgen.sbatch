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

# Source introspection: key-generation code, to find why the dealer's mask is one bit wide
# (p_reveal: low 51 bits of the activations = mask in {0,1}, so `share - mask` is zero).
# Nothing measured or published.
set -uo pipefail

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

SIF="${SIF:-/scratch/tmp/j_wred02/snni_campaign/llama/sif/llama-online-o0_default.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

apptainer exec "$SIF" bash -c '
R=/opt/EzPC/GPU-MPC/ext/sytorch
echo "=== where random_ge lives ==="
grep -rn "random_ge" $R --include=*.h --include=*.cpp --include=*.impl 2>/dev/null | head -20

echo
echo "=== its definition ==="
for f in $(grep -rl "random_ge" $R --include=*.h --include=*.cpp 2>/dev/null | head -6); do
    awk "/random_ge/,/^}/" "$f" | head -30 | sed "s|^|[$(basename $f)] |"
done

echo
echo "=== initializeInferencePartyA / B ==="
grep -rn -A25 "void initializeInferencePartyA" $R --include=*.h 2>/dev/null | head -40
echo "---"
grep -rn -A25 "void initializeInferencePartyB" $R --include=*.h 2>/dev/null | head -40

echo
echo "=== what seeds the mask PRNG ==="
grep -rn "prngs\[\|prng.random\|aes_init\|osuCrypto::PRNG\|SetSeed\|prng_seed" $R --include=*.h --include=*.cpp 2>/dev/null | head -20
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
