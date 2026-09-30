#!/bin/bash
#SBATCH --job-name=llextract
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/llextract-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/llextract-%j.err

# Copies one header out of the measured image for a step overlay; overlay path = in-image path.
set -uo pipefail

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

B=/scratch/tmp/j_wred02/snni_campaign/llama
SIF="${SIF:-$B/sif/llama-online-o0_default.sif}"
# WHICH header, so the same job serves any file the overlay needs a modified copy of.
SRC="${SRC:-/opt/EzPC/GPU-MPC/ext/sytorch/include/sytorch/layers/layers.h}"
OUT="${OUT:-$B/build/extracted/$(basename "$SRC")}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

mkdir -p "$(dirname "$OUT")"
apptainer exec "$SIF" cat "$SRC" > "$OUT" || exit 3
[ -s "$OUT" ] || { echo "FAILED: extracted header is empty" >&2; exit 3; }

# A per-file marker that cannot be right by accident, so a wrong extraction fails here, not at build.
case "$(basename "$SRC")" in
    layers.h) grep -q "this->wseed = ++g_weightSeedCtr" "$OUT" \
                  || { echo "FAILED: no per-layer weight seed in layers.h" >&2; exit 4; } ;;
    module.h) grep -q "allNodesInExecutionOrder" "$OUT" \
                  || { echo "FAILED: no execution-order loop in module.h" >&2; exit 4; } ;;
esac

echo "extracted $(wc -l < "$OUT") lines to $OUT"
md5sum "$OUT"
exit 0
