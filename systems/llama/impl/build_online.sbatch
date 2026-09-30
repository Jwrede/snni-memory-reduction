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
# OVERRIDABLE so a diagnostic overlay (with instrumentation that must never reach a published
# image) can be built under its own name instead of overwriting o0_default.sif.
OUT="${OUT:-$B/sif/llama-online-o0_default.sif}"
OVERLAY="${OVERLAY:-$B/build/overlay}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

[ -r "$PARENT" ] || { echo "no parent image at $PARENT" >&2; exit 2; }
[ -d "$OVERLAY" ] || { echo "no overlay at $OVERLAY" >&2; exit 2; }

def=$B/def/$(basename "$OUT" .sif).def
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
    $OVERLAY /opt/patch

%post
    set -e
    test -d /opt/patch
    cp -r /opt/patch/. /opt/EzPC/GPU-MPC/ext/sytorch/
    rm -rf /opt/patch
    /opt/rebuild_step.sh

%labels
    campaign snni-memory-reduction
    system llama
    step o0_default
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

# AND THE STEP'S OWN LEVER, when named: the overlay checks above pass for every step regardless of
# whether ITS change applied, which would otherwise let a no-op step publish its predecessor's peak
# under its own name. Opt-in via LEVER_FILE/LEVER_PATTERN, checked against the FINISHED IMAGE.
if [ -n "${LEVER_PATTERN:-}" ]; then
    LEVER_FILE="${LEVER_FILE:?LEVER_PATTERN given without LEVER_FILE}"
    # -F: literal match, since lever source text (e.g. `delete[] rp;`) is full of regex syntax.
    HITS=$(apptainer exec "$OUT" grep -cF -- "$LEVER_PATTERN" "$LEVER_FILE" 2>/dev/null || echo 0)
    MIN="${LEVER_MIN:-1}"
    if [ "$HITS" -lt "$MIN" ]; then
        echo "BUILD_FAIL: this step's lever is not in the image. Expected at least $MIN match(es)"
        echo "            of '$LEVER_PATTERN' in $LEVER_FILE, found $HITS. The image would have"
        echo "            measured its predecessor under this step's name."
        exit 6
    fi
    echo "lever check: $HITS match(es) of '$LEVER_PATTERN' in $LEVER_FILE"
fi
echo "BUILD_OK"
