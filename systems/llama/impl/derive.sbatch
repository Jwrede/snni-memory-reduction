#!/bin/bash
#SBATCH --job-name=llderive
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/llama/derive-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/llama/derive-%j.out
#
#   sbatch --export=ALL,STEP=d0_default derive.sbatch
#
# One step's raw run -> canonical object tables. On the cluster (symbolisation needs the step's image).
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/llama
# RUN=<n> selects one replicate. Every replicate is derived, not just the median one, since the
# others test that the object RANKING (not just the peak height) is stable.
R=$B/results/$STEP${RUN:+/run$RUN}
# Only one of the two polled parties' directories is the published peak; `party0.pid` names it.
P=$(ls -d "$R"/poll_* 2>/dev/null | head -1)
if [ -r "$R/party0.pid" ]; then
    _p0=$(cat "$R/party0.pid")
    [ -d "$R/poll_$_p0" ] && P=$R/poll_$_p0
fi
# freeze_mode=maxparty (tools/palma/measure.sh, 2026-09-29): the party holding the maximum over
# the parties is captured with every party stopped, so the heavier party's own capture is frozen at
# its peak and is the one decomposed. Echoes that party's poll directory; returns 1 for any other
# run, and also when the heavier party has no capture (its peak a spike the poller never saw as
# the maximum), which is said, and the default choice stands.
maxparty_dir() {   # <run dir>
    local run="$1" d best="" bk=0 kb
    grep -q '^freeze_mode=maxparty$' "$run/RUN_META" 2>/dev/null || return 1
    for d in "$run"/poll_*; do
        [ -d "$d" ] || continue
        kb=$(awk 'BEGIN{m=0}
                  /^#/{ if (match($0,/hwm_kb=[0-9]+/)) { v=substr($0,RSTART+7,RLENGTH-7)+0; if (v>m) m=v }; next }
                  { if ($4+0 > m) m=$4+0 }
                  END{print m}' "$d/vmrss.log" 2>/dev/null)
        [ "${kb:-0}" -gt "$bk" ] && { bk="$kb"; best="$d"; }
    done
    [ -n "$best" ] || return 1
    if [ ! -s "$best/peaks/resident.txt" ]; then
        echo "NOTE: freeze_mode=maxparty but the heavier party $(basename "$best") has no capture;" \
             "the default party choice stands" >&2
        return 1
    fi
    echo "freeze_mode=maxparty: decomposing the heavier party $(basename "$best") (VmHWM $bk kB)" >&2
    cat "$best/peaks/CAPTURE" >&2 2>/dev/null
    echo "$best"
}
MP=$(maxparty_dir "$R") && P="$MP"
# SIF overridable so a probe run made with another step's image derives against the right one.
SIF="${SIF:-$B/sif/llama-online-$STEP.sif}"

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
