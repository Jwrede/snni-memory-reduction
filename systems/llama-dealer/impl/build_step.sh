#!/bin/bash
# Build one step's image from its own Dockerfile (in that step's impl/), separately from the
# measurement run so no compile competes with the measured peak.  build_step.sh <STEP>
set -uo pipefail
STEP="${1:?need STEP}"
BASE=~/snni_campaign/llama-dealer
CTX="$BASE/steps/$STEP/impl"

[ -f "$CTX/Dockerfile" ] || { echo "ERROR: no Dockerfile at $CTX" >&2; exit 2; }

podman image exists localhost/llama-dealer-base || {
    echo "building the shared base first"
    podman build -f "$BASE/Dockerfile.base" -t llama-dealer-base "$BASE" || exit 1
}

podman build -f "$CTX/Dockerfile" -t "llama-dealer-$STEP" "$CTX" || exit 1

# The digest is this step's provenance, recorded as impl_fingerprint in steps.csv.
podman inspect --format '{{.Id}}' "localhost/llama-dealer-$STEP:latest" \
    > "$BASE/steps/$STEP/impl/IMAGE_DIGEST"
echo "built llama-dealer-$STEP  $(cat "$BASE/steps/$STEP/impl/IMAGE_DIGEST")"
