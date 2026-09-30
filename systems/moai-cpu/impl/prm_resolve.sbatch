#!/bin/bash
#SBATCH --job-name=prmresolve
#SBATCH --partition=express
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/prm_resolve-%j.out
#
#   sbatch --export=ALL,STEP=p_prm0 prm_resolve.sbatch
#
# Resolves both tables of one recorder run (pmrec: pool growth; pool recorder: live at the peak) with
# the same resolver, maps and --program-root.
set -uo pipefail
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
W=/scratch/tmp/j_wred02/snni_campaign
STEP="${STEP:?need STEP}"
R=$W/moai-cpu/results/$STEP/run1
SIF=$W/moai-cpu/sif/moai-cpu-$STEP.sif
P=$R/poll/peaks
MAPS=$P/pm.maps.vaddr; [ -r "$MAPS" ] || MAPS=$P/pm.maps
[ -r "$SIF" ] || { echo "FAILED: kein Image $SIF" >&2; exit 2; }

echo "############ $STEP  ($(cat $R/PEAK.txt 2>/dev/null | tr "\n" " "))"
echo
echo "=== pmrec: Pool-WACHSTUM am Peak ==="
head -1 $P/resident.txt
python3 $W/attrib_resident/resolve_resident.py "$P/resident.txt" "$MAPS" \
    "$R/prm_objects_pmrec.jsonl" --exec "apptainer exec $SIF" --top 15 --program-root /root/moai
echo
echo "=== Pool-Recorder: LEBENDE Objekte am selben Peak ==="
head -1 $P/resident_pool.txt
MAPS2=$P/pool.maps; [ -r "$MAPS2" ] || MAPS2="$MAPS"
python3 $W/attrib_resident/resolve_resident.py "$P/resident_pool.txt" "$MAPS2" \
    "$R/prm_objects_pool.jsonl" --exec "apptainer exec $SIF" --top 20 --program-root /root/moai
