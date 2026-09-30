#!/bin/bash
#SBATCH --job-name=moai-gpu-linkbuild
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/diag/linkbuild-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/diag/linkbuild-%j.err
#
# Diagnostic. Builds the pristine source with static and dynamic cudart (static link blocks
# LD_PRELOAD interposition of cudaMallocAsync), inside the existing image, to test the peak effect.
set -uo pipefail

BASE=/scratch/tmp/j_wred02/snni_campaign/moai-gpu
DIAG=$BASE/diag
SRC=$DIAG/src-pristine
SIF=/scratch/tmp/j_wred02/moai_gpu/moai-gpu-s0_default.sif

mkdir -p "$DIAG" || exit 2
[ -r "$SIF" ] || { echo "no image at $SIF" >&2; exit 2; }
[ -d "$SRC" ] || { echo "no source at $SRC (untar it first)" >&2; exit 2; }

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

export APPTAINER_CACHEDIR=$DIAG/appcache
export APPTAINER_TMPDIR=$DIAG/apptmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "=== source state ==="
ls "$SRC"

# Both preconditions checked here: the first attempt built the untouched tree (RMSNorm unit test,
# no BERT layer), which ran and "passed" in 14 seconds with markers compiled into nothing.
NMARK=$(grep -c 'snni_mem_marker(' "$SRC/src/include/test/test_single_layer.cuh" 2>/dev/null || echo 0)
[ "$NMARK" -ge 16 ] || { echo "FAILED: source is not instrumented ($NMARK markers)" >&2; exit 3; }
echo "markers in source: $NMARK"

LIVE=$(grep -E '^\s*[a-zA-Z_]+_test\(\);' "$SRC/src/test.cu" | tr -d ' ')
[ "$LIVE" = "single_layer_test();" ] || {
    echo "FAILED: test.cu selects [$LIVE], expected single_layer_test();" >&2
    echo "        run select_workload.py on the source tree first" >&2
    exit 3
}
echo "workload selected: $LIVE"
# The header is pulled in by include.cuh; test.cu's own commented include is stale and wrong.
grep -q '^#include "test/test_single_layer.cuh"' "$SRC/src/include/include.cuh" \
    || { echo "FAILED: include.cuh does not pull in test/test_single_layer.cuh" >&2; exit 3; }

build_one() {
    # build_one <builddir> <extra cmake args...>
    local bdir="$1"; shift
    echo
    echo "############ building $bdir  extra: $* ############"
    apptainer exec --bind /scratch:/scratch "$SIF" bash -lc "
        set -e
        export HOME=$DIAG/home; mkdir -p \$HOME
        cd $SRC
        cmake -S . -B $bdir -DCMAKE_BUILD_TYPE=Release $*
        cmake --build $bdir -j\$(nproc)
    "
    return $?
}

RC=0
build_one build-static || RC=$?
build_one build-shared -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared || RC=$?

echo
echo "################################################################"
echo "### THE CHECK THAT SAYS WHETHER THIS EXPERIMENT TESTS ANYTHING"
echo "### Two binaries that link the same way would make the comparison"
echo "### vacuous, and a vacuous comparison passes silently."
echo "################################################################"
for b in build-static build-shared; do
    BIN=$SRC/$b/test
    echo
    echo "--- $b ---"
    if [ ! -x "$BIN" ]; then echo "MISSING: $BIN"; RC=4; continue; fi
    ls -l "$BIN"
    echo "md5: $(md5sum "$BIN" | awk '{print $1}')"
    apptainer exec --bind /scratch:/scratch "$SIF" bash -lc "
        echo -n 'NEEDED cudart: '; readelf -d $BIN | grep -c libcudart || true
        echo -n 'cudaMallocAsync defined in binary: '; nm --defined-only $BIN 2>/dev/null | grep -c ' T cudaMallocAsync' || true
        echo -n 'cudaMallocAsync undefined (interposable): '; nm -D -u $BIN 2>/dev/null | grep -c cudaMallocAsync || true
        echo 'readelf -d NEEDED:'; readelf -d $BIN | grep NEEDED
    "
done

echo
echo "=== EXPECTED: build-static has NEEDED cudart 0 and defined 1;"
echo "===           build-shared has NEEDED cudart 1 and undefined 1."
echo "=== Anything else and the two builds are not what they claim to be."
echo "rc=$RC"
exit $RC
