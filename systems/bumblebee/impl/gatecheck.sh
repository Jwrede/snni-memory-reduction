#!/bin/bash
# Determinism check: two unchanged runs, gate observable compared byte for byte (login node; submits only).
#
#   gatecheck.sh [step_id]        step_id defaults to b0_default
#
# Empirical, since the randomness is inside SPU/yacl. Runs the step's own image as
# results/<step>_gatecheck; policy from campaign.env.
set -uo pipefail

STEP="${1:-b0_default}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/bumblebee

. "$C/harness/campaign.env" || exit 1

POLICY="SNNI_PM_MIN=$SNNI_PM_MIN,SNNI_PM_FULLSCAN=$SNNI_PM_FULLSCAN"
POLICY="$POLICY,SNNI_PM_FREEZE=$SNNI_PM_FREEZE,POLL_MS=$POLL_MS"
POLICY="$POLICY,SNNI_POLICY_VERSION=$SNNI_POLICY_VERSION"

SIF=$B/sif/bumblebee-$STEP.sif
[ -r "$SIF" ] || { echo "no image at $SIF" >&2; exit 2; }

cd "$B" || exit 1
if [ -d "$B/results/${STEP}_gatecheck" ]; then
    mv "$B/results/${STEP}_gatecheck" "$B/results/.superseded-${STEP}_gatecheck-$(date -u +%Y%m%d-%H%M%S)"
fi

prev=""
for r in 1 2; do
    dep=""
    [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    # shellcheck disable=SC2086
    j=$(sbatch --parsable $dep --job-name="bbgate$r" \
        --export="ALL,STEP=${STEP}_gatecheck,RUN=$r,SIF=$SIF,$POLICY,PMREC=1" \
        bumblebee.sbatch) || exit 1
    echo "gatecheck run$r job=$j${prev:+  (after $prev)}"
    prev="$j"
done
echo "compare with: cmp results/${STEP}_gatecheck/run1/gate.txt results/${STEP}_gatecheck/run2/gate.txt"
