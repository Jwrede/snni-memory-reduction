#!/bin/bash
#SBATCH --job-name=moaigpu-img
# Partition express: a normal-partition build waited ~10.5 h (14.08) for minutes of work.
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
# 24G = 2.7x the measured need (8.74/8.41 GB); 96G fit only one express node (95000 MB each).
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --exclude=r09n16
# r09n16 excluded: two builds hung at "Extracting local image..." (apptainer fakeroot, futex_do_wait).
# 30 min limit: a build takes ~2.5 min; hangs were otherwise indistinguishable from slow builds.
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/img-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/img-%x-%j.err
#
# Builds one step's image (digest recorded in steps.csv).
#
#   sbatch --export=ALL,STEP=s0_default,SRC=<dir> build_step_image.sbatch
#
# Baseline = pristine upstream + markers/workload/seed+gate (identical at every step); a step adds its
# lever. Non-default build flags: CMAKE_CUDA_RUNTIME_LIBRARY=Shared (device recorder interposition;
# indistinguishable from static, systems/moai-gpu/linkmode.md) and -g (call sites; no residency cost).
set -uo pipefail

B=/scratch/tmp/j_wred02/snni_campaign/moai-gpu
STEP="${STEP:?need STEP}"
SRC="${SRC:?need SRC, a prepared source tree}"
BASE_SIF="${BASE_SIF:-/scratch/tmp/j_wred02/moai_gpu/moai-gpu-s0_default.sif}"
OUT=$B/sif/moai-gpu-$STEP.sif

[ -d "$SRC" ] || { echo "no source at $SRC" >&2; exit 2; }
[ -r "$BASE_SIF" ] || { echo "no base image at $BASE_SIF" >&2; exit 2; }
mkdir -p "$B/sif" "$B/imgtmp" "$B/imgcache" || exit 2

# Refuse to overwrite. An image silently replaced under a step id is the same failure class as a
# results directory silently merged into its predecessor.
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists. Move it aside." >&2; exit 3; }

# Source must carry all three changes; a build quietly lacking one runs and measures the wrong thing.
NMARK=$(grep -c 'snni_mem_marker(' "$SRC/src/include/test/test_single_layer.cuh" 2>/dev/null || echo 0)
[ "$NMARK" -ge 16 ] || { echo "FAILED: source has $NMARK markers, need 16" >&2; exit 4; }
grep -q 'GATE|out|' "$SRC/src/include/test/test_single_layer.cuh" || { echo "FAILED: no gate in source" >&2; exit 4; }
grep -q 'snni_prng' "$SRC/thirdparty/phantom-fhe/include/prng.cuh" || { echo "FAILED: seed not pinned" >&2; exit 4; }
# Both draw sources checked separately: pinning random_bytes alone leaves adjust_sk_hamming_weight's
# unseeded shuffle still varying the secret key per run (measured 2026-08-02).
grep -q 'snni_sk_mx' "$SRC/thirdparty/phantom-fhe/src/secretkey.cu" || { echo "FAILED: the hamming-weight shuffle is not pinned (second draw source)" >&2; exit 4; }
LIVE=$(grep -E '^\s*[a-zA-Z_]+_test\(\);' "$SRC/src/test.cu" | tr -d ' ')
[ "$LIVE" = "single_layer_test();" ] || { echo "FAILED: test.cu selects [$LIVE]" >&2; exit 4; }
echo "source checks: markers=$NMARK gate=yes seed=yes workload=$LIVE"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
export APPTAINER_TMPDIR=$B/imgtmp APPTAINER_CACHEDIR=$B/imgcache

DEF=$B/imgtmp/moai-gpu-$STEP.def
cat > "$DEF" <<EOF
Bootstrap: localimage
From: $BASE_SIF

%files
    $SRC /moai-campaign

%post
    set -e
    rm -rf /moai
    mv /moai-campaign /moai
    cd /moai
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \\
        -DCMAKE_CXX_FLAGS=-g -DCMAKE_CUDA_FLAGS=-g
    cmake --build build -j\$(nproc)
    test -x /moai/build/test

%environment
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH
EOF

echo "=== def ==="; cat "$DEF"
echo "=== building ==="
apptainer build "$OUT" "$DEF"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "BUILD FAILED rc=$RC." >&2
    echo "If this is an unprivileged-build restriction rather than a compile error, the fallback" >&2
    echo "is to compile onto /scratch and run the base image against that binary, recording" >&2
    echo "binary_md5 as the step fingerprint instead of an image digest. Say so in the MANIFEST" >&2
    echo "rather than letting a step's provenance quietly become weaker than every other system's." >&2
    exit "$RC"
fi

echo "=== what was built ==="
ls -l "$OUT"
echo "image_digest=$(sha256sum "$OUT" | awk '{print $1}')"
apptainer exec "$OUT" bash -lc '
    BIN=/moai/build/test
    echo "binary_md5=$(md5sum $BIN | awk "{print \$1}")"
    echo -n "NEEDED cudart (must be 1): "; readelf -d $BIN | grep -c libcudart
    echo -n "cudaMallocAsync interposable (must be 1): "; nm -D -u $BIN | grep -c cudaMallocAsync
    echo -n "debug_info sections (must be > 0): "; readelf -S $BIN | grep -c debug_info
    echo -n "single_layer_test present: "; nm $BIN | grep -c single_layer_test
'
exit 0
