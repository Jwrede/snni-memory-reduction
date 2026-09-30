#!/bin/bash
#SBATCH --job-name=scpuderive
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/shaft-cpu/derive-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/shaft-cpu/derive-%x-%j.out
#
# One replicate -> canonical object table (compute node; addr2line needs the step's image). Every
# replicate derived (ranking stability). 32G (8G OOM'd once).
#
#   sbatch --export=ALL,STEP=s0_default,RUN=1 derive.sbatch
set -uo pipefail
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/shaft-cpu
STEP="${STEP:?need STEP}"
R=$B/results/$STEP${RUN:+/run$RUN}
# ONE IMAGE SERVES EVERY STEP: this system is env-gated, so symbolisation always needs the same
# image's libraries. See the note in shaft_cpu_campaign.sbatch.
SIF="${SIF:-$B/sif/shaft-cpu-s0_default.sif}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }
[ -d "$R" ] || { echo "FAILED: no run at $R"; exit 1; }
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 1; }

# apptainer doesn't auto-bind /scratch; without this the extractors (which must run inside the
# image) are invisible and only the python half quietly reports "No such file".
BIND="--bind /scratch/tmp/j_wred02:/scratch/tmp/j_wred02"

# ---------------------------------------------------------------------------------------------
# Decomposes the freeze leader, not the taller party: only the leader's capture is atomic (the
# follower's race has produced attribution figures over 1000% elsewhere). Parties agree to 0.14%
# here, so the atomic-but-slightly-lower leader is used, with both peaks printed for visibility.
party_of() {   # <poll dir> -> party number, or empty
    local pid="${1##*/poll_}" pf
    for pf in "$R"/party*.pid; do
        [ -f "$pf" ] || continue
        [ "$(tr -dc '0-9' < "$pf")" = "$pid" ] && { basename "$pf" .pid | tr -dc 0-9; return; }
    done
}
peak_of() {    # <poll dir> -> VmHWM kB, falling back to the polled maximum
    local kb
    kb=$(awk -F= '/^# hwm_kb=/{print $2}' "$1/vmrss.log" 2>/dev/null)
    [ -n "${kb:-}" ] && [ "$kb" != 0 ] || kb=$(awk -F= '/^# peak_kb=/{print $2}' "$1/vmrss.log" 2>/dev/null)
    echo "${kb:-0}"
}

BEST=""; MAXKB=0
echo "=== parties in this run"
for d in "$R"/poll_*; do
    [ -d "$d" ] || continue
    kb=$(peak_of "$d")
    if grep -q "poller on pid ${d##*/poll_} (leader=1)" "$R/poller_attach.log" 2>/dev/null; then
        lead=yes; BEST="$d"
    else
        lead=no
    fi
    [ "$kb" -gt "$MAXKB" ] && MAXKB="$kb"
    pnum=$(party_of "$d")
    echo "  party=${pnum:-?} trace=$(basename "$d") peak_kb=$kb freeze_leader=$lead"
done
# No leader trace (freeze disabled, or a single party): fall back to the highest peak, and say so.
if [ -z "$BEST" ]; then
    echo "NOTE: no freeze-leader trace found; falling back to the highest-peak trace."
    for d in "$R"/poll_*; do
        [ -d "$d" ] || continue
        [ "$(peak_of "$d")" = "$MAXKB" ] && { BEST="$d"; break; }
    done
fi
[ -n "$BEST" ] || { echo "FAILED: no poll trace under $R"; exit 1; }
LEADER=yes
# Under freeze_mode=maxparty (tools/palma/measure.sh) the party holding the maximum over the parties
# is captured with every party stopped, so the heavier party's last capture is frozen at its own
# peak: the published (heavier) party is decomposed, with its own census. A heavier party with no
# capture (its peak a spike the poller never saw as the maximum) leaves the leader's, and says so.
if grep -q '^freeze_mode=maxparty$' "$R/RUN_META" 2>/dev/null; then
    for d in "$R"/poll_*; do
        [ -d "$d" ] || continue
        if [ "$(peak_of "$d")" = "$MAXKB" ]; then
            if [ -s "$d/peaks/resident.txt" ]; then
                [ "$d" = "$BEST" ] || LEADER=no
                BEST="$d"
            else
                echo "NOTE: the heavier party $(basename "$d") has no capture; decomposing the leader"
            fi
            break
        fi
    done
    echo "freeze_mode=maxparty: decomposing $(basename "$BEST") (freeze leader: $LEADER)"
    cat "$BEST/peaks/CAPTURE" 2>/dev/null
fi
BESTKB=$(peak_of "$BEST")
PARTY=$(party_of "$BEST")
echo "decomposing party=${PARTY:-unknown} trace=$(basename "$BEST") peak_kb=$BESTKB"
echo "published peak over parties=$MAXKB kB; decomposed party is $BESTKB kB"
awk -v a="$BESTKB" -v b="$MAXKB" 'BEGIN{if(b>0)printf "party gap = %.3f%% of the published peak\n",100*(b-a)/b}'

for f in "$BEST/peaks/resident.txt" "$BEST/peaks/pm.maps"; do
    [ -s "$f" ] || { echo "FAILED: missing or empty $f"; exit 1; }
done

echo "=== scan cost actually paid (samples, mean, max)"
awk '{s+=$3; n++; if ($3>m) m=$3}
     END {if (n) printf "samples=%d  mean=%.0f ms  max=%.0f ms\n", n, s/n, m}' \
    "$BEST/peaks/PM_INDEX" 2>/dev/null || echo "(no PM_INDEX)"

echo "=== naming the call sites"
apptainer exec $BIND "$SIF" python3 "$C/attrib_resident/resolve_resident.py" \
    "$BEST/peaks/resident.txt" "$BEST/peaks/pm.maps" "$R/objects_resident.jsonl" \
    --exec "apptainer exec $BIND $SIF" --top 20 || exit 1

# Heap side uses the Python census where available (same reasoning as ARION's Go heap dump): the
# recorder sees the memory but can't name it (81% of peak is torch's CPU allocator, unfixable by
# frame depth). Recorder's table stays as reconciliation (95% coverage at last measurement).
HEAPSRC="$R/objects_resident.jsonl"
CENSUS=$(ls "$BEST/../pycensus_pid"*.txt 2>/dev/null | head -1)
[ -z "$CENSUS" ] && CENSUS=$(ls "$R/pycensus_pid"*.txt 2>/dev/null | head -1)
if grep -q '^freeze_mode=maxparty$' "$R/RUN_META" 2>/dev/null; then
    CENSUS="$R/pycensus_pid${BEST##*/poll_}.txt"
fi
if [ -s "$CENSUS" ]; then
    PEAKKB=$(awk '/polled_peak_kb=/{split($0,a,"="); print a[2]}' "$R/PEAK.txt" 2>/dev/null | head -1)
    echo "=== the Python census names the heap side ($CENSUS)"
    python3 "$C/attrib_pycensus/pycensus_objects.py" "$CENSUS" "$R/objects_pycensus.jsonl" \
        ${PEAKKB:+--peak-kb "$PEAKKB"} && HEAPSRC="$R/objects_pycensus.jsonl"
else
    echo "=== no Python census in this run: the heap side stays the recorder's, which on this"
    echo "    system names the allocator and not an object"
fi

echo "=== assembling the host decomposition"
apptainer exec $BIND "$SIF" python3 "$C/smaps_objects.py" "$BEST/peaks" \
    "$R/objects_host.jsonl" --heap "$HEAPSRC" || exit 1

echo "DERIVE_OK party=${PARTY:-unknown} leader=$LEADER"
