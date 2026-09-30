#!/bin/bash
#SBATCH --job-name=scpubuild
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/shaft-cpu/build-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/shaft-cpu/build-%x-%j.out
#
# docker archive -> .sif; checks that the recorder loads inside it. Job, not login node.
#
#   sbatch --export=ALL,TAR=/scratch/.../shaft-cpu-s0_default.tar,NAME=s0_default build_sif.sbatch
set -uo pipefail
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/shaft-cpu
AR=$C/attrib_resident
TAR="${TAR:?need TAR=<docker-archive tar>}"
NAME="${NAME:?need NAME=<step id>}"
SIF="$B/sif/shaft-cpu-$NAME.sif"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }
mkdir -p "$B/sif"
[ -s "$TAR" ] || { echo "FAILED: no archive at $TAR"; exit 1; }

# Build somewhere with space, not in $HOME: apptainer's cache and temp for a multi-GB image are
# both larger than the home quota here.
export APPTAINER_CACHEDIR="$B/.apptainer-cache"
export APPTAINER_TMPDIR="$B/.apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if [ -e "$SIF" ]; then
    echo "REFUSING: $SIF exists. Move it aside rather than overwriting: an image is what a"
    echo "published number was measured with."
    exit 3
fi

echo "=== building $SIF from $TAR"
apptainer build "$SIF" "docker-archive://$TAR" || exit 1
sha256sum "$SIF"

# ---------------------------------------------------------------------------------------------
# Proves the recorder loads before any measurement depends on it: a trivial program is not
# sufficient, since the TLS recursion guard's behavior depends on how many shared objects are
# loaded, so import torch (which has killed it before) is checked too.
echo
echo "=== recorder: does the container's loader accept it"
apptainer exec --bind "$AR":/ar --bind /scratch/tmp/j_wred02:/scratch/tmp/j_wred02 "$SIF" \
    bash -c 'LD_PRELOAD=/ar/pmrec.so SNNI_PM_TABLE=/tmp/pmt_%d SNNI_PM_MIN=65536 \
             /bin/true && echo "TRIVIAL_OK"' || { echo "RECORDER_FAILED trivial"; exit 2; }

echo
echo "=== recorder: does it survive import torch (the case that has killed it before)"
apptainer exec --bind "$AR":/ar --bind /scratch/tmp/j_wred02:/scratch/tmp/j_wred02 "$SIF" \
    bash -c 'LD_PRELOAD=/ar/pmrec.so SNNI_PM_TABLE=/tmp/pmt_%d SNNI_PM_MIN=65536 \
             python3 -c "import torch, crypten, transformers; print(\"TORCH_OK\", torch.__version__)"' \
    || { echo "RECORDER_FAILED import torch"; exit 2; }

echo
echo "=== the image runs the payload at all (no poller, no recorder)"
apptainer exec "$SIF" python3 -c "import crypten, transformers, torch; print('IMPORTS_OK')" || exit 2

echo "BUILD_OK $SIF"
