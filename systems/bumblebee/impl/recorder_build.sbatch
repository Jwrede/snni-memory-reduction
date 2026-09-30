#!/bin/bash
#SBATCH --job-name=bbrecbuild
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/bumblebee/recbuild-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/bumblebee/recbuild-%j.out
#
#   sbatch --export=ALL,STEP=b0_default recorder_build.sbatch
#
# Rebuilds the shared resident recorder (adds aligned_alloc, memalign, C++17 over-aligned/nothrow
# operator new) against glibc 2.28 (one artifact for every image). Previous artifacts moved aside
# with a timestamp.
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
AR=$C/attrib_resident
B=$C/bumblebee
SIF=$B/sif/bumblebee-$STEP.sif

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 1; }

export APPTAINER_TMPDIR=$B/apptmp
export APPTAINER_CACHEDIR=$B/apptmp/cache
mkdir -p "$APPTAINER_CACHEDIR"

STAMP=$(date -u +%Y%m%d-%H%M%S)
for f in pmrec.so pmsample; do
    [ -e "$AR/$f" ] && cp -p "$AR/$f" "$AR/.superseded-$f-$STAMP"
done
echo "previous artifacts preserved as $AR/.superseded-*-$STAMP"

echo "=== source the fix is being built from"
grep -c 'aligned_alloc' "$AR/pmrec.c"
md5sum "$AR/pmrec.c"

echo
echo "=== building against glibc 2.28, then asking THIS image's loader to accept it"
bash "$AR/build_in_image.sh" --builder docker://gcc:9-buster "$SIF" "$AR" || {
    echo "BUILD_FAIL: recorder build"; exit 2; }

echo
echo "=== the new library must export the entry points the old one lacked"
for sym in aligned_alloc memalign malloc calloc realloc posix_memalign _Znwm _Znam; do
    if apptainer exec --bind "$AR":/ar "$SIF" bash -c "nm -D /ar/pmrec.so | grep -q ' T $sym\$'"; then
        echo "  intercepts $sym"
    else
        echo "  MISSING    $sym"
        MISS=1
    fi
done
[ -z "${MISS:-}" ] || { echo "BUILD_FAIL: the rebuilt recorder does not export what it must"; exit 3; }

ls -la "$AR/pmrec.so" "$AR/pmsample"
echo RECBUILD_OK
