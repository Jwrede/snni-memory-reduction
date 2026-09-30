#!/bin/bash
#SBATCH --job-name=sgderive
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/shaft-gpu/derive-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/shaft-gpu/derive-%x-%j.out
#
# One step's three runs -> objects_vram.jsonl (torch snapshot) and objects_host.jsonl (smaps + pmrec).
# Compute node. 32G (8G OOM'd once).
#
#   sbatch --export=ALL,STEP=s1_limb_loop derive.sbatch
set -uo pipefail
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/shaft-gpu
STEP="${STEP:?need STEP}"
SIF="${SIF:-/scratch/tmp/j_wred02/shaft_gpu/shaft-gpu-estream.sif}"
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
# apptainer doesn't auto-bind /scratch; without this the extractors (which must run inside the
# image) are invisible and only the python half quietly reports "No such file".
BIND="--bind /scratch/tmp/j_wred02:/scratch/tmp/j_wred02"

# One directory per replicate (matching the measurement sbatch): previously all runs of a step
# shared one directory, so a derive had no single replicate to be about.
CLEAN=$B/results/$STEP${RUN:+/run$RUN}
# Device snapshot comes from the measured run itself since 2026-08-01 (SNNI_COMBINED=1), falling
# back to a separate _attrib channel if present, so a missing snapshot is never silent.
DEV=$B/results/${STEP}_attrib${RUN:+/run$RUN}
[ -d "$DEV" ] || DEV=$CLEAN
# The host recorder now rides along with the measured run (see conf/shaft-gpu.conf), so the
# resident table and the peak come from ONE execution. This used to read a separate _pmrec run.
HOST=$CLEAN
[ -d "$CLEAN" ] || { echo "ERROR: no clean run at $CLEAN" >&2; exit 2; }

# --- device pool ------------------------------------------------------------------------------
SNAP=$(ls -S "$DEV"/*.pickle "$DEV"/**/*.pickle 2>/dev/null | head -1)
if [ -n "${SNAP:-}" ]; then
    apptainer exec $BIND "$SIF" python3 "$C/torch_peak_objects.py" "$SNAP" "$CLEAN/objects_vram.jsonl" \
        && echo "wrote objects_vram.jsonl from $(basename "$SNAP")"
else
    echo "WARNING: no torch snapshot under $DEV; this step has no device decomposition" >&2
fi

# --- host pool --------------------------------------------------------------------------------
# Decomposition is taken from the same party whose peak is published (the larger of the two),
# picked by party identity (run_gpu.sh's party<N>.pid) rather than by independently taking the
# max on each side, which could pair one party's smaps with the other's objects.
party_of() {  # <run dir> <poll dir> -> party number, or empty
    local run="$1" pid="${2##*/poll_}" pf
    for pf in "$run"/n12_seq128/party*.pid; do
        [ -f "$pf" ] || continue
        [ "$(cat "$pf" 2>/dev/null)" = "$pid" ] && { basename "$pf" .pid | tr -dc 0-9; return; }
    done
}
poll_of() {   # <run dir> <party> -> poll dir, or empty
    local run="$1" want="$2" d
    for d in "$run"/poll_*; do
        [ -d "$d" ] || continue
        [ "$(party_of "$run" "$d")" = "$want" ] && { echo "$d"; return; }
    done
}

BIGGEST=""; BIGKB=0; BIGPARTY=""
for d in "$CLEAN"/poll_*; do
    [ -d "$d" ] || continue
    kb=$(awk -F= '/peak_kb/{print $2}' "$d/vmrss.log" 2>/dev/null)
    [ -n "${kb:-}" ] && [ "$kb" -gt "$BIGKB" ] && { BIGKB="$kb"; BIGGEST="$d"; }
done
[ -n "$BIGGEST" ] || { echo "ERROR: no poll trace in $CLEAN" >&2; exit 2; }
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
if MP=$(maxparty_dir "$CLEAN"); then
    BIGGEST="$MP"; BIGKB=$(awk -F= '/peak_kb/{print $2}' "$MP/vmrss.log" 2>/dev/null)
fi
BIGPARTY=$(party_of "$CLEAN" "$BIGGEST")
echo "clean host peak $BIGKB kB in $(basename "$BIGGEST") (party ${BIGPARTY:-unknown})"

RES=""
if [ -n "$BIGPARTY" ]; then
    RES=$(poll_of "$HOST" "$BIGPARTY")
    [ -n "$RES" ] && [ -f "$RES/peaks/resident.txt" ] || {
        echo "WARNING: party $BIGPARTY has no resident table in the attribution run" >&2; RES=""; }
    [ -n "$RES" ] && echo "resident table from the SAME party ($BIGPARTY): $(basename "$RES")"
else
    echo "WARNING: could not identify the peak party; falling back to the largest resident" >&2
    RESKB=0
    for d in "$HOST"/poll_*; do
        [ -f "$d/peaks/resident.txt" ] || continue
        kb=$(awk -F= '/peak_kb/{print $2}' "$d/vmrss.log" 2>/dev/null)
        [ -n "${kb:-}" ] && [ "$kb" -gt "$RESKB" ] && { RESKB="$kb"; RES="$d"; }
    done
fi

if [ -n "$RES" ]; then
    apptainer exec $BIND "$SIF" python3 "$C/attrib_resident/resolve_resident.py" \
        "$RES/peaks/resident.txt" "$RES/peaks/pm.maps" "$CLEAN/objects_resident.jsonl" \
        --exec "apptainer exec $BIND $SIF" --top 20 || echo "WARNING: symbol resolution failed" >&2
    apptainer exec $BIND "$SIF" python3 "$C/smaps_objects.py" "$BIGGEST/peaks" \
        "$CLEAN/objects_host.jsonl" --heap "$CLEAN/objects_resident.jsonl" \
        && echo "wrote objects_host.jsonl [pagemap+smaps]"
else
    echo "WARNING: no resident table under $HOST; falling back to libraries only" >&2
    apptainer exec $BIND "$SIF" python3 "$C/smaps_objects.py" "$BIGGEST/peaks" "$CLEAN/objects_host.jsonl"
fi
echo "DONE $STEP"
