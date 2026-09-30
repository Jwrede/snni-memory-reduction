#!/bin/bash
#SBATCH --job-name=llderive
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/llama-dealer/derive-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/llama-dealer/derive-%j.out
#
#   sbatch --export=ALL,STEP=d0_default derive.sbatch
#
# One step's raw run -> canonical object tables. On the cluster (addr2line needs the step's image).
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/llama-dealer
# RUN=<n> selects one replicate; every replicate is derived (not just the median) since the
# spread covers peak height but not whether object ranking is stable.
R=$B/results/$STEP${RUN:+/run$RUN}
P=$R/poll_dealer
# SIF may be overridden, so a probe run made with another step's image can be
# derived against the image it actually ran.
SIF="${SIF:-$B/sif/llama-dealer-$STEP.sif}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }

for f in "$P/peaks/resident.txt" "$P/peaks/pm.maps" "$SIF"; do
    [ -s "$f" ] || { echo "FAILED: missing or empty $f"; exit 1; }
done

echo "=== scan cost actually paid (ts rss scan_ms)"
awk '{s+=$3; n++; if ($3>m) m=$3}
     END {if (n) printf "samples=%d  mean=%.0f ms  max=%.0f ms\n", n, s/n, m}' \
    "$P/peaks/PM_INDEX" 2>/dev/null || echo "(no PM_INDEX)"

echo "=== naming the call sites"
python3 "$C/attrib_resident/resolve_resident.py" \
    "$P/peaks/resident.txt" "$P/peaks/pm.maps" "$R/objects_resident.jsonl" \
    --exec "apptainer exec $SIF" --top 15 || exit 1

echo "=== assembling the host decomposition"
python3 "$C/smaps_objects.py" "$P/peaks" "$R/objects_host.jsonl" \
    --heap "$R/objects_resident.jsonl" || exit 1

echo "DERIVE_OK"
