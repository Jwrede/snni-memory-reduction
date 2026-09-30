#!/bin/bash
#SBATCH --job-name=markerresolve
#SBATCH --partition=express
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/marker_resolve-%j.out
#
#   sbatch --export=ALL,STEP=p_prm2 marker_resolve.sbatch
#
# Resolves the per-marker live tables of one run with one resolver, maps and --program-root.
# Reason: the peak is a plateau (layer 1 within 0.122%, three markers equal to the kB); the freeze
# lands at different instants per run and names different rank-1 rows; the row holding memory across
# the plateau decides. No new measurement (tables written by poll_peak.sh during the run).
set -uo pipefail
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
W=/scratch/tmp/j_wred02/snni_campaign
STEP="${STEP:?need STEP}"
R=$W/moai-cpu/results/$STEP/run1
P=$R/poll/peaks
OUT=$R/markers_resolved
mkdir -p "$OUT"

# The image comes from the run that was measured, not from the step name: addr2line runs INSIDE it.
SIF=""
[ -r "$R/RUN_META" ] && SIF=$(grep -aoE 'image=[^[:space:]]+' "$R/RUN_META" | head -1 | cut -d= -f2-)
[ -r "${SIF:-}" ] || SIF=$W/moai-cpu/sif/moai-cpu-$STEP.sif
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

MAPS=$P/pool.maps; [ -r "$MAPS" ] || { MAPS=$P/pm.maps.vaddr; [ -r "$MAPS" ] || MAPS=$P/pm.maps; }
echo "############ $STEP  image=$SIF  maps=$MAPS"
echo "############ $(tr '\n' ' ' < "$R/PEAK.txt" 2>/dev/null)"
echo

# Marker order comes from MARKER_INDEX, which is the order the program emitted them in. Sorting the
# filenames would put after_attention before layer_0_begin and read like a different run.
n=0
while read -r ts rss name; do
    [ -n "${name:-}" ] || continue
    # BY TIMESTAMP, NOT BY NAME. `marker_<name>_pool.txt` is keyed on the marker name alone, and
    # both layers emit the same names, so layer 1's sample overwrites layer 0's and twelve of the
    # files hold the wrong layer under a plausible name. The pool trace keeps every sample under
    # its own timestamp (`pool_<ts>.txt`, a hard link), so that is the source of truth here.
    src=$P/pool_${ts}.txt
    [ -s "$src" ] || src=$P/marker_${name}_pool.txt
    [ -s "$src" ] || { echo "--- $name: no pool table"; continue; }
    n=$((n + 1))
    printf -v tag '%02d_%s' "$n" "$name"
    echo "=== $tag   ts=$ts  rss_kb=$rss"
    head -1 "$src"
    python3 "$W/attrib_resident/resolve_resident.py" "$src" "$MAPS" \
        "$OUT/$tag.jsonl" --exec "apptainer exec $SIF" --top 8 --program-root /root/moai
    echo
done < "$P/MARKER_INDEX"

# The near-peak sample itself. It is not a marker, so it is not in MARKER_INDEX; it is the pool
# trace entry closest to the peak, and it is what the published decomposition would have used.
PEAK_TS=$(awk '{print $1}' "$P/PEAK" 2>/dev/null)
BEST=""; BESTD=""
for f in "$P"/pool_*.txt; do
    [ -s "$f" ] || continue
    t=$(basename "$f" .txt); t=${t#pool_}
    d=$(( t > PEAK_TS ? t - PEAK_TS : PEAK_TS - t ))
    [ -z "$BESTD" ] || [ "$d" -lt "$BESTD" ] && { BESTD=$d; BEST=$f; }
done
if [ -n "$BEST" ]; then
    echo "=== nearest sample to the peak: $(basename "$BEST")  (${BESTD} ms from PEAK ts $PEAK_TS)"
    head -1 "$BEST"
    python3 "$W/attrib_resident/resolve_resident.py" "$BEST" "$MAPS" \
        "$OUT/99_nearest_peak.jsonl" --exec "apptainer exec $SIF" --top 8 --program-root /root/moai
fi
echo "wrote $OUT"
