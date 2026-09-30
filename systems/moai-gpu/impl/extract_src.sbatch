#!/bin/bash
#SBATCH --job-name=mgsrc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/mgsrc-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/mgsrc-%j.err

# Copies the measured source tree out of the baseline image (base for patched step builds). Job, not
# login node.
set -uo pipefail
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

BASE_SIF="${BASE_SIF:-/scratch/tmp/j_wred02/moai_gpu/moai-gpu-s0_default.sif}"
OUT="${OUT:-/scratch/tmp/j_wred02/snni_campaign/moai-gpu/src/pristine}"
[ -r "$BASE_SIF" ] || { echo "FAILED: no base image at $BASE_SIF" >&2; exit 2; }
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists. Move it aside." >&2; exit 3; }

mkdir -p "$(dirname "$OUT")" || exit 2
apptainer exec --bind "$(dirname "$OUT")":"$(dirname "$OUT")" "$BASE_SIF" \
    bash -c "cp -a /moai '$OUT' && rm -rf '$OUT/build'" || exit 3

# Three things that cannot be right by accident, so a wrong extraction fails here rather than later.
[ -s "$OUT/thirdparty/phantom-fhe/include/fft.h" ] \
    || { echo "FAILED: no fft.h in the extracted tree" >&2; exit 4; }
[ -s "$OUT/thirdparty/phantom-fhe/src/ckks.cu" ] \
    || { echo "FAILED: no ckks.cu in the extracted tree" >&2; exit 4; }
grep -q "snni_mem_marker(" "$OUT/src/include/test/test_single_layer.cuh" 2>/dev/null \
    || { echo "FAILED: the extracted tree carries no phase markers, so it is not the MEASURED tree" >&2; exit 4; }

echo "extracted to $OUT"
du -sh "$OUT"
find "$OUT" -name "*.cu" -o -name "*.cuh" -o -name "*.h" | wc -l | sed 's/^/source files: /'
exit 0
