#!/bin/bash
# Submits a whole line as one serial dependency chain (every step, replicate, channel).
#
#   run_line.sh <system> <step_id> [<step_id> ...]
#
# Only for levers already chosen (re-measurement, reproduction). New steps: run_step.sh.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SYS="${1:?usage: run_line.sh <system> <step_id> [<step_id> ...]}"
shift
[ $# -gt 0 ] || { echo "no steps given" >&2; exit 1; }

dep=""
for step in "$@"; do
    out=$(SNNI_DEP="$dep" bash "$HERE/run_step.sh" "$SYS" "$step") || {
        echo "$out"
        echo "SUBMISSION FAILED at $step; the rest of the line was NOT submitted" >&2
        echo "steps already submitted keep running: check with squeue before resubmitting" >&2
        exit 1
    }
    echo "$out"
    dep=$(printf '%s\n' "$out" | sed -n 's/^LAST_JOB=//p' | tail -1)
    [ -n "$dep" ] || {
        echo "no job id came back from $step, so the next step cannot be chained behind it; "\
             "stopping rather than submitting steps that would run concurrently" >&2
        exit 1
    }
done
