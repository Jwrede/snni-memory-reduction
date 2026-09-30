#!/bin/bash
# Build the LLAMA dealer line as apptainer images from source on PALMA; run as a SLURM job, not
# on the login node. Each step's def is generated here (identical shape) rather than kept per step.
#   build_line.sh                 # base, then every step in waterfall order
#   build_line.sh d2_mask_stream  # one step (its parent must already exist)
set -uo pipefail

C=/scratch/tmp/j_wred02/snni_campaign
B=$C/llama-dealer
SRC=$B/build          # base_patch/, rebuild_step.sh, steps/<id>/patch/
OUT=$B/sif
# Canonical waterfall order; a step's parent is its predecessor here.
ORDER=(d0_default d1_select_free d2_mask_stream d3_activation_free d4_truncate_free d5_gelu_free)
STEPS=("${ORDER[@]:1}")

# --base-only builds the base image and d0 and stops: a lever may only be chosen after its
# predecessor is measured, so prebuilding the whole line presumes the ranking the measurement produces.
if [ "${1:-}" = "--base-only" ]; then
    STEPS=()
    shift
elif [ $# -gt 0 ]; then
    STEPS=("$@")
fi

parent_of() {
    local want="$1" prev=llama-dealer-base
    for x in "${ORDER[@]}"; do
        [ "$x" = "$want" ] && { echo "$prev"; return 0; }
        [ "$x" = "d0_default" ] || prev="llama-dealer-$x"
    done
    echo "ERROR: $want is not in the waterfall order" >&2
    return 1
}

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
mkdir -p "$OUT" "$B/def"

if [ $# -eq 0 ] && [ ! -f "$OUT/llama-dealer-base.sif" ]; then
    echo "=== base"
    apptainer build --fakeroot "$OUT/llama-dealer-base.sif" "$SRC/llama-base.def" || exit 1
    # d0 is the baseline: the base image itself, under the name the waterfall publishes.
    cp "$OUT/llama-dealer-base.sif" "$OUT/llama-dealer-d0_default.sif"
fi

# ${arr[@]+...} guards the empty --base-only array under set -u on older bash.
for s in ${STEPS[@]+"${STEPS[@]}"}; do
    PARENT=$(parent_of "$s") || exit 2
    [ -d "$SRC/steps/$s/patch" ] || { echo "ERROR: no overlay at $SRC/steps/$s/patch" >&2; exit 2; }
    def="$B/def/$s.def"
    cat > "$def" <<EOF
Bootstrap: localimage
From: $OUT/$PARENT.sif

# $s -- built on top of $PARENT, from the overlay published in this step's impl/patch/.
# rebuild_step.sh touches the tree (so make cannot decide the parent's binary is up to date) and
# fails the build if the resulting binary equals the parent's, because a step that does not change
# the binary cannot measure a lever.

# /opt, not /tmp: apptainer bind-mounts the host /tmp over the container's during %post, so a
# file staged into /tmp by %files is invisible to the very section that consumes it.

%files
    $SRC/steps/$s/patch /opt/patch

%post
    set -e
    test -d /opt/patch
    cp -r /opt/patch/. /opt/EzPC/GPU-MPC/ext/sytorch/
    rm -rf /opt/patch
    /opt/rebuild_step.sh

%labels
    campaign snni-memory-reduction
    system llama-dealer
    step $s
    parent $PARENT
EOF
    echo "=== $s (from $PARENT)"
    apptainer build --fakeroot "$OUT/llama-dealer-$s.sif" "$def" || exit 1
done

echo "=== built:"
ls -la "$OUT"
for f in "$OUT"/*.sif; do
    printf '%-40s %s\n' "$(basename "$f")" \
        "$(apptainer exec "$f" md5sum /opt/EzPC/GPU-MPC/ext/sytorch/build/bertbenchmark | awk '{print $1}')"
done
