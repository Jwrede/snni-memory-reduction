#!/bin/bash
# Move the step images from the podman host to PALMA unchanged (transferred, not rebuilt, so
# binary_md5 is identical on both sides and the machine is the only cross-machine variable).
# One multi-image archive (steps share a ~4.9 GB base layer), streamed through this host.
#   export_images.sh [step ...]        (default: the six published dealer steps)
set -uo pipefail

REMOTE_C="${REMOTE_C:-jwmaster01}"
REMOTE_P="${REMOTE_P:-palma}"
DEST="${DEST:-/scratch/tmp/j_wred02/snni_campaign/llama-dealer/images}"
STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || STEPS=(d0_default d1_select_free d2_mask_stream d3_activation_free
                                d4_truncate_free d5_gelu_free)

IMAGES=()
for s in "${STEPS[@]}"; do IMAGES+=("localhost/llama-dealer-$s"); done
echo "exporting: ${IMAGES[*]}"

ssh "$REMOTE_P" "mkdir -p '$DEST'" || exit 1

# podman save streams to a file on PALMA; digests are taken on both ends (a truncated archive
# would still build a plausible image).
ssh "$REMOTE_C" "podman save --multi-image-archive ${IMAGES[*]}" \
    | ssh "$REMOTE_P" "cat > '$DEST/llama-dealer-steps.tar'" || exit 1

echo "=== size and digest on PALMA:"
ssh "$REMOTE_P" "ls -l '$DEST/llama-dealer-steps.tar'; sha256sum '$DEST/llama-dealer-steps.tar'"
