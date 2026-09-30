#!/bin/bash
#SBATCH --job-name=sharkderive
#SBATCH --partition=zen2-128C-496G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/shark/derive-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/shark/derive-%j.out
#
#   sbatch --export=ALL,STEP=s0_default,RUN=1 derive.sbatch
#
# One run's raw poll trace -> object tables (job; symbolisation needs the step's image).
# objects_host.jsonl = freeze leader's capture (RUN_META freeze_leader_party); follower derived as
# symmetry evidence only.
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/shark
R=$B/results/$STEP${RUN:+/run$RUN}
SIF="${SIF:-$B/sif/shark-$STEP.sif}"
# Dealer line measures the same binary as online, so it falls back to the baseline image, but only
# for probes/refuted-attempts/baseline (p_*, x_*, *_default). A waterfall step must run its own
# image or risk measuring the wrong binary (happened 2026-08-05, s2_threads_4).
if [ ! -r "$SIF" ]; then
    case "${STEP:-}" in
        p_*|x_*|*_default)
            SIF="$B/sif/shark-s0_default.sif" ;;
        *)
            echo "FAILED: step '${STEP:-}' has no image at $SIF, and a waterfall step may not fall" >&2
            echo "        back to the baseline: it would measure the baseline binary under this" >&2
            echo "        step's name. Build its image, or name one explicitly through SIF." >&2
            exit 2 ;;
    esac
fi

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }

[ -s "$SIF" ] || { echo "FAILED: missing image $SIF"; exit 1; }
[ -s "$R/dealer.pid" ] || { echo "FAILED: no $R/dealer.pid, cannot tell which trace is the leader"; exit 1; }
P0PID=$(tr -dc '0-9' < "$R/dealer.pid")
P1PID=$(tr -dc '0-9' < "$R/party1.pid" 2>/dev/null)

# Leading party is read from RUN_META, not guessed, so derive never pairs one party's frozen
# capture with the other's peak. Absent (pre-2026-07-30 runs) means party 0.
LEADP=$(sed -n 's/^freeze_leader_party=//p' "$R/RUN_META" 2>/dev/null | tr -dc '0-9')
LEADP="${LEADP:-0}"
if [ "$LEADP" = 1 ]; then FOLLOWP=0; LEADPID="$P1PID"; FOLLOWPID="$P0PID"
else                      FOLLOWP=1; LEADPID="$P0PID"; FOLLOWPID="$P1PID"; fi

trace_peak() {   # <poll dir> -> max VmHWM (or polled max) in kB
    awk 'BEGIN{m=0}
         /^#/{ if (match($0,/hwm_kb=[0-9]+/)) { v=substr($0,RSTART+7,RLENGTH-7)+0; if (v>m) m=v } ; next }
         { if ($4+0 > m) m=$4+0; if ($2+0 > m) m=$2+0 }
         END{print m}' "$1/vmrss.log" 2>/dev/null
}

# Dealer is one process, so no follower/party comparison. Poll directory name is read from the
# poller's own log line, not rebuilt from the pid: rebuilding it broke six derives in one afternoon.
LEAD=$(sed -n 's/.*(leader=1)[[:space:]]*->[[:space:]]*//p' "$R/poller_attach.log" 2>/dev/null | tail -1)
[ -n "$LEAD" ] || LEAD="$R/poll"
FOLLOW=""
[ -s "$LEAD/vmrss.log" ] || { echo "FAILED: no leader trace at $LEAD"; exit 1; }

PK0=$(trace_peak "$LEAD")
# Empty, deliberately: the symmetry check and the follower decomposition below are both guarded on
# it, so they skip instead of reporting the absence of a second party as a finding about this one.
PK1=""
echo "dealer (the measured role, single process) peak=${PK0} kB"

# Checked, not assumed: if the follower peaks materially above the leader, published peak and
# published decomposition would describe different processes. 1% threshold is measured from
# baseline/replicate spread, not a round number.
TOL_PROMILLE=10
if [ -n "$PK1" ] && [ "$PK0" -gt 0 ] 2>/dev/null; then
    EXC=$(( (PK1 - PK0) * 1000 / PK0 ))
    if [ "$EXC" -ge "$TOL_PROMILLE" ]; then
        echo "FAILED: party $FOLLOWP peaked ${EXC} promille ABOVE the freeze leader (${PK1} > ${PK0} kB)." >&2
        echo "The published peak would then be party $FOLLOWP's while the published decomposition is" >&2
        echo "party $LEADP's, which is the pair-by-party rule being broken. Re-run this step with" >&2
        echo "SHARK_FREEZE_LEADER=$FOLLOWP, or publish it as party-$FOLLOWP-peak with party-$FOLLOWP" >&2
        echo "decomposition, but not the mixture." >&2
        exit 1
    fi
    if [ "$PK1" -gt "$PK0" ]; then
        echo "NOTE: party $FOLLOWP is ${EXC} promille above the freeze leader. Inside the 1% tolerance, so"
        echo "      the published peak (party $FOLLOWP's) and the published decomposition (party $LEADP's,"
        echo "      the atomic capture) differ by less than this step's own replicate spread."
    fi
fi

for f in "$LEAD/peaks/resident.txt" "$LEAD/peaks/pm.maps"; do
    [ -s "$f" ] || { echo "FAILED: missing or empty $f"; exit 1; }
done

echo "=== scan cost actually paid (ts rss scan_ms)"
awk '{s+=$3; n++; if ($3>m) m=$3}
     END {if (n) printf "samples=%d  mean=%.0f ms  max=%.0f ms\n", n, s/n, m}' \
    "$LEAD/peaks/PM_INDEX" 2>/dev/null || echo "(no PM_INDEX)"

echo "=== naming the call sites (party $LEADP, the atomic capture)"
python3 "$C/attrib_resident/resolve_resident.py" \
    "$LEAD/peaks/resident.txt" "$LEAD/peaks/pm.maps" "$R/objects_resident.jsonl" \
    --exec "apptainer exec $SIF" --top 15 || exit 1

echo "=== assembling the host decomposition (party $LEADP)"
python3 "$C/smaps_objects.py" "$LEAD/peaks" "$R/objects_host.jsonl" \
    --heap "$R/objects_resident.jsonl" || exit 1

# Follower is evidence only (non-atomic capture, not read by make_steps.py); output filename
# carries the follower's party number so it never overwrites the published table.
if [ -s "$FOLLOW/peaks/resident.txt" ] && [ -s "$FOLLOW/peaks/pm.maps" ]; then
    echo "=== party $FOLLOWP (follower, non-atomic capture): evidence only"
    python3 "$C/attrib_resident/resolve_resident.py" \
        "$FOLLOW/peaks/resident.txt" "$FOLLOW/peaks/pm.maps" "$R/objects_resident_party$FOLLOWP.jsonl" \
        --exec "apptainer exec $SIF" --top 15 \
        && python3 "$C/smaps_objects.py" "$FOLLOW/peaks" "$R/objects_host_party$FOLLOWP.jsonl" \
            --heap "$R/objects_resident_party$FOLLOWP.jsonl" \
        || echo "NOTE: party $FOLLOWP derivation failed; it is evidence, not a dependency"
fi

echo "DERIVE_OK"
