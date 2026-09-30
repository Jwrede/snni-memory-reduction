#!/bin/bash
#SBATCH --job-name=lobuild
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/llama/build-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/llama/build-%j.out
#
# Builds the LLAMA online baseline image: the dealer line's base image plus the overlay that makes
# the online path runnable and gateable (../patches/). Runs as a job, not on the login node.
set -uo pipefail

C=/scratch/tmp/j_wred02/snni_campaign
B=$C/llama
PARENT=$C/llama-dealer/sif/llama-dealer-base.sif
OUT=$B/sif/llama-online-p_iotrace.sif

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

[ -r "$PARENT" ] || { echo "no parent image at $PARENT" >&2; exit 2; }
[ -d "$B/build/overlay" ] || { echo "no overlay at $B/build/overlay" >&2; exit 2; }

def=$B/def/p_iotrace.def
cat > "$def" <<EOF
Bootstrap: localimage
From: $PARENT

# LLAMA online baseline: the dealer line's base image plus the overlay that makes the online path
# runnable and gateable.
#
# /opt, not /tmp: apptainer bind-mounts the host /tmp over the container's during %post, so a file
# staged into /tmp by %files is invisible to the very section that consumes it. That cost two
# builds which reported COMPLETED and produced nothing.

%files
    $B/build/overlay-probe /opt/patch

%post
    set -e
    test -d /opt/patch
    cp -r /opt/patch/. /opt/EzPC/GPU-MPC/ext/sytorch/
    rm -rf /opt/patch
    /opt/rebuild_step.sh

%labels
    campaign snni-memory-reduction
    system llama
    step p_iotrace (diagnostic: socket IO trace + runtime depth)
    parent llama-dealer-base
EOF

echo "=== building $OUT"
# --force: rebuilding a step must replace its image rather than refuse on a prior artifact.
apptainer build --force --fakeroot "$OUT" "$def" || { echo "BUILD_FAIL"; exit 1; }

# A build that cannot fail is not a check. The artifact is verified, not assumed.
BIN=/opt/EzPC/GPU-MPC/ext/sytorch/build/bertbenchmark
NEW=$(apptainer exec "$OUT" md5sum $BIN | awk '{print $1}')
OLD=$(apptainer exec "$PARENT" md5sum $BIN | awk '{print $1}')
echo "parent binary md5 = $OLD"
echo "online binary md5 = $NEW"
[ "$NEW" != "$OLD" ] || { echo "BUILD_FAIL: binary identical to parent, the overlay did not take"; exit 3; }

# Check that the stock net.load() call is gone, not just the filename (which legitimately
# survives in the patch's own explanatory comment).
SRCF=/opt/EzPC/GPU-MPC/ext/sytorch/examples/bertbenchmark.cpp
if apptainer exec "$OUT" grep -qE '^[[:space:]]*net\.load\(' "$SRCF"; then
    echo "BUILD_FAIL: an active net.load() call is still present; the overlay did not apply"
    exit 4
fi
if ! apptainer exec "$OUT" grep -q 'GATE|logits|' "$SRCF"; then
    echo "BUILD_FAIL: the gate observable is absent from the source in this image"
    exit 5
fi
echo "BUILD_OK"
