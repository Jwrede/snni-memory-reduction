#!/bin/bash
#SBATCH --job-name=moaicpu-t16-requeue
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/requeue-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/requeue-%j.out
#
# K4 (2026-08-03): baseline threads 16. Runs (afterany) when the T=72 chain drains; moves those results
# aside as off-path evidence and submits the T=16 chain.
set -uo pipefail
CAMP=/scratch/tmp/j_wred02/snni_campaign
BASE=$CAMP/moai-cpu
STAMP=$(date -u +%Y%m%d-%H%M%S)

# Belt and braces beyond the dependency: refuse rather than move a directory out from under a writer.
others=$(squeue -u "$USER" -h -n moaicpu -o %i 2>/dev/null | grep -v "^${SLURM_JOB_ID}$" || true)
if [ -n "$others" ]; then
    echo "REFUSING: moaicpu jobs still in the queue: $others" >&2
    exit 7
fi

if [ -d "$BASE/results/m0_default" ]; then
    DEST="$BASE/results/.superseded-m0_default-t72-$STAMP"
    mv "$BASE/results/m0_default" "$DEST" || exit 7
    {
        echo "# Why this measurement set is superseded"
        echo
        echo "Campaign decision 2026-08-03 (PLAN_ABGABE, thread unification): baselines run at"
        echo "T=16 unless the system's own default run sets a thread count, and this system has"
        echo "none -- its 72 was the node's core count, i.e. inherited from the hardware."
        echo
        echo "These runs were measured at OMP_NUM_THREADS=72 and are kept as OFF-PATH evidence"
        echo "for the thread lever: same image, same binary_md5 as the T=16 baseline, only T"
        echo "differs. A completed run here is the measured upper end of the lever the earlier"
        echo "campaign saw as 132.7 -> 90.2 -> 73.1 GiB over 72 -> 32 -> 16 threads."
        echo
        echo "State at requeue time ($STAMP UTC):"
        for r in "$DEST"/run*/; do
            [ -d "$r" ] || continue
            rn=$(basename "$r")
            pk="absent"; [ -s "$r/PEAK.txt" ] && pk="present"
            gt="absent"; [ -s "$r/gate.txt" ] && gt="present"
            ec="?"; [ -s "$r/program_exit_code.txt" ] && ec=$(cat "$r/program_exit_code.txt")
            echo "  $rn: PEAK.txt $pk, gate.txt $gt, program_exit_code=$ec"
        done
        echo
        echo "run1 is a DEAD run, not a replicate: its job (45677202) was requeued by SLURM"
        echo "mid-flight around 13:10 on 2026-08-03 and the second attempt refused its own"
        echo "leftover directory. A run without PEAK.txt is not a replicate."
    } > "$DEST/WHY.md"
    echo "moved m0_default -> $DEST"
else
    echo "no m0_default results directory; nothing to move"
fi

# Submit the T=16 baseline chain through the shared driver; the sbatch on disk pins 16 now.
if ! "$CAMP/harness/run_step.sh" moai-cpu m0_default; then
    echo "NEEDS_MANUAL: run_step.sh failed; submit the T=16 chain by hand" >&2
    touch "$BASE/NEEDS_MANUAL_T16"
    exit 7
fi
echo "T16_CHAIN_SUBMITTED"
