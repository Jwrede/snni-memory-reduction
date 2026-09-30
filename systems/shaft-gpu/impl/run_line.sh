#!/bin/bash
# SHAFT-GPU: runs one waterfall step unattended.  run_line.sh <step> [LEVER=VAL ...]
# Three runs, each awaited to a terminal SLURM state:
#   <step>          clean        the only run whose peaks and wall time may be published
#   <step>_attrib   device       torch state mode, VRAM composition only
#   <step>_pmrec    host         pmrec + pagemap, host composition in RESIDENT bytes
# Failed clean run stops the sequence; attribution failures are not fatal (make_steps.py reports
# incomplete vs absent).
set -uo pipefail

STEP="${1:?need step}"
shift || true
LEVERS=("$@")

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
SB=/scratch/tmp/j_wred02/snni_campaign/shaft-gpu/shaft_gpu_campaign.sbatch
: "${SNNI_MAX_WAIT_S:=21600}"
: "${SNNI_POLL_S:=30}"
export SNNI_MAX_WAIT_S SNNI_POLL_S

run() {  # <suffix> <extra export vars...>
    local suffix="$1"; shift
    local name="$STEP$suffix"
    echo "=== $name  $(date -u +%FT%TZ)"
    timeout $((SNNI_MAX_WAIT_S + 600)) \
        bash "$REPO/tools/palma_step.sh" "$SB" "$name" "${LEVERS[@]}" "$@"
}

run "" || { echo "clean run failed for $STEP; not attributing a run that did not happen" >&2; exit 1; }
run "_attrib" ATTRIB=1 || echo "WARNING: device attribution failed for $STEP" >&2
run "_pmrec"  PMREC=1  || echo "WARNING: host attribution failed for $STEP" >&2
echo "=== $STEP done  $(date -u +%FT%TZ)"
