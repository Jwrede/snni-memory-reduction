#!/bin/bash
#SBATCH --job-name=moaigpu-modraise
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/probe-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/probe-%j.out

# Source introspection of Bootstrapper::modraise_inplace (19.8% at s1's VRAM peak; rank 1
# check<cudaError> 53.6% is an inlined wrapper). Nothing measured or published.
set -uo pipefail
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }
SIF=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/sif/moai-gpu-s1_complexroots_free.sif
[ -r "$SIF" ] || SIF=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/sif/moai-gpu-s0_default.sif
[ -r "$SIF" ] || { echo "FAILED: no image" >&2; exit 2; }
echo "image: $SIF"

apptainer exec "$SIF" bash -c '
R=/moai/src
# A file only the right tree can have, not just an existing directory (a CMake compiler-probe
# directory walked straight through that weaker check earlier today).
[ -f "$R/test.cu" ] || { echo "FAILED: no source tree at $R"; exit 3; }

F=$R/include/source/bootstrapping/Bootstrapper.cu
echo "=== modraise_inplace in full ==="
S=$(grep -n "void Bootstrapper::modraise_inplace" "$F" | head -1 | cut -d: -f1)
[ -n "$S" ] || { echo "FAILED: modraise_inplace not found in $F"; exit 3; }
E=$(awk -v s="$S" "NR>s && /^}/ { print NR; exit }" "$F")
echo "lines $S..$E"
sed -n "${S},${E}p" "$F"

echo
echo "=== who calls it, and how often per layer ==="
grep -rn "modraise_inplace" $R --include=*.cu --include=*.cuh --include=*.hpp | grep -v "void Bootstrapper::" | head -10

echo
echo "=== what a PhantomCiphertext costs, i.e. what one live copy is worth ==="
grep -rn "class PhantomCiphertext" -A25 /moai/thirdparty/phantom-fhe/include/ciphertext.h 2>/dev/null | grep -nE "cuda_auto_ptr|size_|chain_index|coeff_modulus" | head -12
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
