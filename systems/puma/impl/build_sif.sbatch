#!/bin/bash
#SBATCH --job-name=puma-img
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/puma/img-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/puma/img-%x-%j.err
#
#   sbatch --export=ALL,STEP=s0_default build_sif.sbatch
#
# Builds as a job (login-node apptainer exec once ran 40+ min). Partition express (pip + one model
# download). Refuses to overwrite an existing image.
set -uo pipefail

B=/scratch/tmp/j_wred02/snni_campaign/puma
STEP="${STEP:?need STEP}"
OUT=$B/sif/puma-$STEP.sif
DEF=$B/puma.def

[ -r "$DEF" ] || { echo "no def at $DEF" >&2; exit 2; }
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists. Move it aside." >&2; exit 3; }
mkdir -p "$B/sif" "$B/imgtmp" "$B/imgcache" || exit 2

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }
export APPTAINER_TMPDIR=$B/imgtmp APPTAINER_CACHEDIR=$B/imgcache

echo "=== building $OUT from $DEF on $(hostname) at $(date -u +%FT%TZ)"
apptainer build "$OUT" "$DEF"
RC=$?
[ "$RC" = 0 ] || { echo "FAILED: apptainer build rc=$RC" >&2; exit "$RC"; }

# Check the image rather than assume it: a build can exit 0 with a missing driver directory.
apptainer exec "$OUT" python3 -c "import spu, jax, flax, transformers; print('imports ok')" || exit 4
apptainer exec "$OUT" test -r /opt/puma/examples/python/utils/nodectl.py || {
    echo "FAILED: nodectl.py missing from the image" >&2; exit 4; }
apptainer exec "$OUT" test -r /opt/puma/examples/python/ml/flax_bert/3pc.json || {
    echo "FAILED: 3pc.json missing from the image" >&2; exit 4; }
apptainer exec "$OUT" python3 -c "
from transformers import AutoTokenizer, FlaxBertForSequenceClassification
AutoTokenizer.from_pretrained('bert-base-uncased')
FlaxBertForSequenceClassification.from_pretrained('bert-base-uncased')
print('model cached offline ok')" || exit 4

echo "=== versions in the image"
apptainer exec "$OUT" cat /opt/puma_versions.txt
apptainer exec "$OUT" cat /opt/puma_spu_commit.txt
echo "sha256 $(sha256sum "$OUT" | awk '{print $1}')"
echo "BUILD_OK $OUT"
