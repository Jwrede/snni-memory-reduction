#!/bin/bash
#SBATCH --job-name=arion-goheap
#SBATCH --partition=bigsmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/arion/goheap-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/arion/goheap-%j.out
#
# ARION census run (M15): object table for the peak (the C recorder cannot see Go allocations).
# Output: (address, size, name) at a frozen instant, joined with the pagemap into resident bytes per
# allocating function (go1.7 dumps carry no per-object type). Separate diagnostic run; capture RSS
# published beside the baseline peak.
# Sequence: SIGSTOP -> pagemap+smaps snapshot -> SIGCONT -> SIGUSR1 (dump under the runtime's own
# stop-the-world). Both RSS values recorded; the join publishes the gap. Dump (heap-sized) parsed
# in-job by seeking, then deleted.
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
BASE=$CAMP/arion
RES=$BASE/results/p_goheap
SIF=$BASE/sif/arion-p_goheap.sif
RUNNER=$BASE/run_arion.sh
TOOLS=$BASE/attrib_go

# 95% of the smallest published baseline replicate (573,724,148 kB): near enough to the peak on a
# monotone-growth system.
TRIGGER_KB=${TRIGGER_KB:-545000000}

[ -r "$SIF" ] || { echo "no census image at $SIF" >&2; exit 2; }
[ -r "$RUNNER" ] || { echo "no runner at $RUNNER" >&2; exit 2; }
[ -d "$TOOLS" ] || { echo "no attrib_go tools at $TOOLS" >&2; exit 2; }
[ -e "$RES" ] && { echo "REFUSING: $RES exists. Move it aside." >&2; exit 3; }
mkdir -p "$RES/Output"

df -B1G "$BASE" | tail -1
avail_gb=$(df -B1G --output=avail "$BASE" | tail -1 | tr -d ' ')
if [ "$avail_gb" -lt 700 ]; then
    echo "REFUSING: only ${avail_gb} GB free under $BASE; the dump needs ~600 GB transiently" >&2
    exit 2
fi

export ARION_N_LAYERS=2 ARION_THREADS=64 ARION_MENU=12 GOGC=100
# Recorder floor comes from campaign.env, never a local copy, so this diagnostic never silently
# drifts from the recorder floor of the measurements it explains.
export SNNI_PM_MIN="${SNNI_PM_MIN:?campaign.env must supply the recorder floor}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

{
    echo "step=p_goheap (diagnostic census, not a step)"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "slurm_job_id=$SLURM_JOB_ID"
    echo "image=$SIF"
    echo "image_digest=$(sha256sum "$SIF" | awk '{print $1}')"
    echo "trigger_kb=$TRIGGER_KB"
    echo "arion_n_layers=$ARION_N_LAYERS arion_threads=$ARION_THREADS arion_menu=$ARION_MENU gogc=$GOGC"
    echo "memprofile_rate=SNNI_PM_MIN=$SNNI_PM_MIN (the campaign recorder floor)"
    echo "godebug=unset, deliberately: the census must see the same page-release behaviour the"
    echo "  baseline had, so madvise semantics are NOT changed here"
} > "$RES/RUN_META"

T0=$(date +%s)
INNER="export ARION_RESULTS=$RES ARION_N_LAYERS=$ARION_N_LAYERS ARION_THREADS=$ARION_THREADS"
INNER="$INNER ARION_MENU=$ARION_MENU GOGC=$GOGC SNNI_PM_MIN=$SNNI_PM_MIN"
INNER="$INNER SNNI_HEAPDUMP_PATH=$RES/heap.dump; exec bash $RUNNER"

apptainer exec \
    --bind "$RES":"$RES" \
    --bind "$RUNNER":"$RUNNER" \
    --bind "$RES/Output":/root/arion/Output \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

PID=""
for _ in $(seq 1 3000); do
    PID=$(cat "$RES/arion.pid" 2>/dev/null || true)
    [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null && break
    kill -0 "$APP_PID" 2>/dev/null || break
    sleep 0.1
done
[ -n "$PID" ] || { echo "ERROR: no payload pid appeared" >&2; wait "$APP_PID"; exit 4; }
echo "payload: $PID"

grep -q "SNNI_HEAPDUMP|armed" "$RES/stderr.log" || {
    sleep 5
    grep -q "SNNI_HEAPDUMP|armed" "$RES/stderr.log" || {
        echo "FAILED: the adapter did not announce itself; this binary would ignore the signal" >&2
        kill "$APP_PID" 2>/dev/null
        exit 5
    }
}

TRIGGERED=0
while kill -0 "$PID" 2>/dev/null; do
    RSS=$(awk '/VmRSS/{print $2}' "/proc/$PID/status" 2>/dev/null || echo 0)
    if [ "${RSS:-0}" -ge "$TRIGGER_KB" ] && [ "$TRIGGERED" = 0 ]; then
        TRIGGERED=1
        echo "TRIGGER at rss_kb=$RSS ($(date -u +%H:%M:%S))"
        kill -STOP "$PID"
        for _ in $(seq 1 100); do
            st=$(awk '{print $3}' "/proc/$PID/stat" 2>/dev/null || echo ?)
            [ "$st" = "T" ] && break
            sleep 0.05
        done
        RSS_FROZEN=$(awk '/VmRSS/{print $2}' "/proc/$PID/status")
        HWM_FROZEN=$(awk '/VmHWM/{print $2}' "/proc/$PID/status")
        echo "frozen: rss_kb=$RSS_FROZEN hwm_kb=$HWM_FROZEN"
        echo "$RSS_FROZEN" > "$RES/rss_at_snapshot_kb.txt"
        echo "$HWM_FROZEN" > "$RES/hwm_at_snapshot_kb.txt"
        cp "/proc/$PID/smaps" "$RES/snapshot.smaps" 2>/dev/null || true
        T_SNAP0=$(date +%s)
        python3 "$TOOLS/pagemap_snapshot.py" "$PID" "$RES/pagemap.bin" | tee -a "$RES/census.log"
        T_SNAP1=$(date +%s)
        echo "snapshot_s=$((T_SNAP1-T_SNAP0))"
        kill -CONT "$PID"
        RSS_AT_SIGNAL=$(awk '/VmRSS/{print $2}' "/proc/$PID/status")
        echo "$RSS_AT_SIGNAL" > "$RES/rss_at_dump_kb.txt"
        kill -USR1 "$PID"
        echo "USR1 sent at rss_kb=$RSS_AT_SIGNAL"
    fi
    sleep 5
done

RC=0; wait "$APP_PID" || RC=$?
T1=$(date +%s)
echo "program rc=$RC wall_s=$((T1-T0)) triggered=$TRIGGERED"
echo "$RC" > "$RES/program_exit_code.txt"

[ "$TRIGGERED" = 1 ] || { echo "FAILED: trigger threshold never reached; no census was taken" >&2; exit 6; }
DUMP=$(ls "$RES"/heap.dump.* 2>/dev/null | head -1)
[ -n "$DUMP" ] || { echo "FAILED: no heap dump was written" >&2; exit 6; }
ls -la "$DUMP"

python3 "$TOOLS/parse_heapdump.py" "$DUMP" "$RES/objects_go.jsonl" | tee -a "$RES/census.log" || exit 7
python3 "$TOOLS/join_residency.py" "$RES/objects_go.jsonl" "$RES/pagemap.bin" \
    "$RES/objects_goheap.jsonl" "$(cat "$RES/rss_at_snapshot_kb.txt")" \
    | tee -a "$RES/census.log" || exit 7

sha256sum "$DUMP" > "$RES/heapdump.sha256"
rm -f "$RES"/heap.dump.*
rm -f "$RES/objects_go.jsonl.tmp" 2>/dev/null
gzip -f "$RES/objects_go.jsonl"
echo "GOHEAP_CENSUS_OK"
