#!/bin/bash
# Submit an entire line as one dependency chain, strictly serial across steps and replicates.
# The payload writes ~47 GB per run to a shared filesystem: concurrent writers corrupt key files
# (no material while exiting 0; truncation to 1.9 GB of 23 GB) and make wall time meaningless.
#   run_line.sh d0_default d1_select_free ...
set -uo pipefail
B=/scratch/tmp/j_wred02/snni_campaign/llama-dealer
cd "$B" || exit 1
prev=""
for step in "$@"; do
    rm -rf "$B/results/$step"
    for r in 1 2 3; do
        dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
        # shellcheck disable=SC2086
        j=$(sbatch --parsable $dep \
            --export=ALL,STEP=$step,PMREC=1,FULLSCAN=1,RUN=$r,POLL_MS=15,SNNI_PM_FREEZE=1 \
            llama_dealer.sbatch) || exit 1
        echo "$step run$r = $j"
        prev="$j"
    done
    for r in 1 2 3; do
        d=$(sbatch --parsable --dependency=afterany:"$prev" \
            --export=ALL,STEP="$step",RUN="$r" derive.sbatch) || exit 1
        echo "$step derive$r = $d"
    done
done
