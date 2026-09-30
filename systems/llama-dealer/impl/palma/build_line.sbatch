#!/bin/bash
#SBATCH --job-name=llbuild
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/llama-dealer/build-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/llama-dealer/build-%j.out
#
# Builds LLAMA dealer images on a compute node (login-node apptainer builds left D-state processes).
#
#   sbatch --export=ALL,STEPS=--base-only build_line.sbatch
#   sbatch --export=ALL,STEPS=d1_select_free build_line.sbatch
#
# Checks that each image exists after the build (predecessor reported COMPLETED for 45448605 and
# 45448821 without an image).
set -uo pipefail

B=/scratch/tmp/j_wred02/snni_campaign/llama-dealer
export APPTAINER_TMPDIR=/scratch/tmp/j_wred02/apptainer_tmp
export APPTAINER_CACHEDIR=/scratch/tmp/j_wred02/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# Loaded here too since a child bash's module load doesn't survive back to this shell; without
# it, the verification block below misreported a missing apptainer as a broken build.
module purge 2>/dev/null
module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }

STEPS="${STEPS:---base-only}"
echo "=== building: $STEPS"

# shellcheck disable=SC2086
bash "$B/build/build_line.sh" $STEPS
rc=$?
echo "build_line.sh rc=$rc"
[ "$rc" -eq 0 ] || { echo "BUILD FAILED rc=$rc"; exit "$rc"; }

# A zero exit is necessary and not sufficient: the artifact has to be there and has to be a real
# image. Checked here rather than trusted, for the same reason as above.
echo "=== artifacts"
ls -la "$B/sif" || exit 1
shopt -s nullglob
sifs=("$B"/sif/*.sif)
[ "${#sifs[@]}" -gt 0 ] || { echo "BUILD FAILED: no .sif produced"; exit 1; }
for f in "${sifs[@]}"; do
    sz=$(stat -c %s "$f")
    [ "$sz" -gt 10000000 ] || { echo "BUILD FAILED: $f is only $sz bytes"; exit 1; }
    apptainer exec "$f" test -x /opt/EzPC/GPU-MPC/ext/sytorch/build/bertbenchmark \
        || { echo "BUILD FAILED: cannot confirm bertbenchmark in $f (apptainer rc=$?)"; exit 1; }
    # Debug info is what makes an allocation site a NAME instead of the nearest exported symbol,
    # so a stripped image would make every step of this line publishable only as inferred.
    apptainer exec "$f" readelf -S /opt/EzPC/GPU-MPC/ext/sytorch/build/bertbenchmark \
        | grep -q debug_info \
        || { echo "BUILD FAILED: $f carries no debug info"; exit 1; }
    printf '%-44s %12s  %s\n' "$(basename "$f")" "$sz" \
        "$(apptainer exec "$f" md5sum /opt/EzPC/GPU-MPC/ext/sytorch/build/bertbenchmark | awk '{print $1}')"
done

echo "BUILD_OK"
