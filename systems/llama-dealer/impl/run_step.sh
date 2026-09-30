#!/bin/bash
# LLAMA dealer, outer launcher on server C (jwmaster01). Runs the step's own prebuilt image; only
# the environment varies at run time.  run_step.sh <STEP> [ATTRIB=1] [OMP_NUM_THREADS=16] ...
# Results -> ~/snni_campaign/llama-dealer/results/<STEP>/; ~47 GB key scratch on /data2, deleted after the md5 gate.
set -uo pipefail
STEP="${1:?need STEP}"; shift || true
BASE=~/snni_campaign/llama-dealer
CAMP=~/snni_campaign
RES=$BASE/results/$STEP
WORK=/data2/llama_dealer_work

# An attribution run uses the same step image; the _pmrec suffix names the mode, not a build.
IMG_STEP="${STEP%_pmrec}"
IMAGE="localhost/llama-dealer-$IMG_STEP"
podman image exists "$IMAGE" || { echo "ERROR: $IMAGE not built; run build_step.sh $IMG_STEP" >&2; exit 2; }
DIGEST=$(podman inspect --format '{{.Id}}' "$IMAGE")

mkdir -p "$RES" "$WORK"
ENVARGS=( -e "STEP_IMAGE=$IMAGE" -e "STEP_IMAGE_DIGEST=$DIGEST" )
for kv in "$@"; do ENVARGS+=( -e "$kv" ); done
echo "image: $IMAGE  $DIGEST"

# Run from a private copy and mount that: bash reads a script by byte offset, so editing it mid-run
# executes a mangled line (it once emptied a completed run's peak field). The copy also archives the version used.
cp "$BASE/inner_campaign.sh" "$RES/inner_campaign.sh"

podman run --rm \
  -v "$WORK":/work \
  -v "$RES":/results \
  -v "$CAMP/poll_peak.sh":/camp/poll_peak.sh:ro \
  -v "$CAMP/attrib_resident":/ar:ro \
  -v "$RES/inner_campaign.sh":/opt/inner_campaign.sh:ro \
  "${ENVARGS[@]}" \
  --entrypoint bash "$IMAGE" /opt/inner_campaign.sh "$STEP"
