#!/bin/bash
# Recorder-floor stability probe (login node; submits only).
#
#   floorprobe.sh <step_id> <floor_bytes>      e.g. floorprobe.sh b0_default 32768
#
# Criterion: halve the floor until the top-object order is stable, `drops` = 0, peak unchanged.
# Results under results/<step>_floor<N>; floor passed explicitly (not from campaign.env).
set -uo pipefail

STEP="${1:?usage: floorprobe.sh <step_id> <floor_bytes>}"
FLOOR="${2:?usage: floorprobe.sh <step_id> <floor_bytes>}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/bumblebee

. "$C/harness/campaign.env" || exit 1

# Everything except the floor comes from campaign.env unchanged, so the probe differs from the
# published run in exactly one variable.
POLICY="SNNI_PM_FULLSCAN=$SNNI_PM_FULLSCAN,SNNI_PM_FREEZE=$SNNI_PM_FREEZE,POLL_MS=$POLL_MS"
POLICY="$POLICY,SNNI_POLICY_VERSION=$SNNI_POLICY_VERSION,SNNI_PM_MIN=$FLOOR"

SIF=$B/sif/bumblebee-$STEP.sif
[ -r "$SIF" ] || { echo "no image at $SIF" >&2; exit 2; }
NAME="${STEP}_floor$FLOOR"

cd "$B" || exit 1
[ -d "$B/results/$NAME" ] && \
    mv "$B/results/$NAME" "$B/results/.superseded-$NAME-$(date -u +%Y%m%d-%H%M%S)"

j=$(sbatch --parsable --job-name="bbfl$FLOOR" \
    --export="ALL,STEP=$NAME,RUN=1,SIF=$SIF,$POLICY,PMREC=1" bumblebee.sbatch) || exit 1
echo "floor probe at $FLOOR bytes: job=$j"
d=$(sbatch --parsable --dependency=afterany:"$j" --job-name="bbfld$FLOOR" \
    --export="ALL,STEP=$NAME,RUN=1,SIF=$SIF,$POLICY" derive.sbatch) || exit 1
echo "derive: job=$d"
echo "then compare the top objects of results/$NAME/run1/objects_host.jsonl against the"
echo "published run's, and check that 'drops=' stayed 0 in its resident.txt."
