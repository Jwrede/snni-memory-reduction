#!/bin/bash
# Runs one step of one system on PALMA and waits for a terminal SLURM state.
#
#   palma_step.sh <sbatch-path> <step> [VAR=VAL ...]
#
# Environment:
#   SNNI_MAX_WAIT_S   hard ceiling on the wait, default 43200 (12 h); exceeded -> non-zero exit
#   SNNI_POLL_S       poll interval, default 60
#   SNNI_REMOTE       ssh host, default palma
# One status line per poll.
set -uo pipefail

SBATCH="${1:?need sbatch path}"
STEP="${2:?need step}"
shift 2

REMOTE="${SNNI_REMOTE:-palma}"
MAX_WAIT="${SNNI_MAX_WAIT_S:-43200}"
POLL="${SNNI_POLL_S:-60}"

EXPORT="ALL,STEP=$STEP"
for kv in "$@"; do EXPORT="$EXPORT,$kv"; done

JOB=$(ssh "$REMOTE" "sbatch --parsable --job-name='${STEP:0:12}' --export='$EXPORT' '$SBATCH'" 2>&1)
case "$JOB" in
    ''|*[!0-9]*) echo "ERROR: sbatch did not return a job id: $JOB" >&2; exit 2 ;;
esac
echo "submitted $STEP as job $JOB"

# sacct is authoritative once the job leaves the queue; squeue only knows about live jobs. Asking
# squeue alone would report "gone" the instant it finishes and lose the exit state entirely.
state_of() {
    local s
    s=$(ssh "$REMOTE" "sacct -j $1 --format=State --noheader --parsable2 2>/dev/null | head -1" 2>/dev/null)
    s="${s%%|*}"; s="${s%% *}"
    [ -n "$s" ] && { echo "$s"; return; }
    ssh "$REMOTE" "squeue -j $1 -h -o %T 2>/dev/null" 2>/dev/null
}

WAITED=0
while [ "$WAITED" -lt "$MAX_WAIT" ]; do
    ST=$(state_of "$JOB")
    case "$ST" in
        COMPLETED)
            echo "job $JOB COMPLETED after ${WAITED}s"; exit 0 ;;
        FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED|REVOKED|SPECIAL_EXIT)
            echo "job $JOB terminal state $ST after ${WAITED}s" >&2; exit 3 ;;
        PENDING|RUNNING|CONFIGURING|COMPLETING|RESIZING|REQUEUED|SUSPENDED|"")
            printf '  %5ss  %s %s\n' "$WAITED" "$JOB" "${ST:-unknown}" ;;
        *)
            echo "  ${WAITED}s $JOB unrecognised state '$ST', still waiting" ;;
    esac
    sleep "$POLL"
    WAITED=$((WAITED + POLL))
done

echo "ERROR: job $JOB still not terminal after ${MAX_WAIT}s; leaving it queued and giving up" >&2
exit 4
