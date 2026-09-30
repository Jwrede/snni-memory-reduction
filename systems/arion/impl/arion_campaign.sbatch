#!/bin/bash
#SBATCH --job-name=arion
#SBATCH --partition=bigsmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/arion/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/arion/slurm-%x-%j.err
#
# ARION (Lattigo/Go, pure FHE) on the shared campaign harness.
# Partition bigsmp (72-core Skylake AVX-512, 1.5 TB) required: earlier campaign measured a 242 GiB
# working-set floor and an OOM kill on a 251 GiB machine. --exclusive and pinned threads required
# (thread count: floor 242 -> 139 GiB from 64 to 16 threads).
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
AR=$CAMP/attrib_resident
BASE=$CAMP/arion

. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy

STEP="${STEP:?need STEP}"
IMG_STEP="${STEP%_pmrec}"; IMG_STEP="${IMG_STEP%_attrib}"; IMG_STEP="${IMG_STEP%_goheap}"
# The Go heap census runs the SAME image as the step it decomposes; only the channel differs.
# THE CENSUS TRIGGER IS DERIVED FROM THIS STEP'S OWN PEAK, not a baseline-calibrated constant: a
# fixed threshold could only decompose steps that never lowered the peak below it (measured
# 2026-08-07, job 45896053: a1_btpeval_shallowcopy ran 3h45m to that failure). Derived instead as
# 95% of the clean run's own PEAK.txt, within the procedure's 10% snapshot-distance limit.
_clean_peak=""
if [ -r "$BASE/results/$IMG_STEP/run${RUN:-1}/PEAK.txt" ]; then
    _clean_peak=$(awk -F= '/^polled_peak_kb=/{print $2}' \
        "$BASE/results/$IMG_STEP/run${RUN:-1}/PEAK.txt" 2>/dev/null | tail -1)
fi
if [ -n "$_clean_peak" ] && [ "$_clean_peak" -gt 0 ] 2>/dev/null; then
    TRIGGER_KB="${TRIGGER_KB:-$(( _clean_peak * 95 / 100 ))}"
    echo "trigger derived from $IMG_STEP run${RUN:-1}: peak=$_clean_peak -> TRIGGER_KB=$TRIGGER_KB"
else
    TRIGGER_KB="${TRIGGER_KB:-545000000}"
    echo "WARNING: no clean-run peak for $IMG_STEP run${RUN:-1}; TRIGGER_KB falls back to the" \
         "baseline-calibrated $TRIGGER_KB, which is only right for the baseline itself"
fi

SIF="${SIF:-$BASE/sif/arion-$IMG_STEP.sif}"
# THE GO ADAPTER HAS TO BE COMPILED IN (no LD_PRELOAD equivalent in Go), so the census channel
# runs its own per-step image carrying the SIGUSR1/WriteHeapDump handler plus MemProfileRate.
# The image name is derived from the step and must exist; a default of `arion-p_goheap.sif` for
# every step would silently decompose the baseline program under the step's name (caught
# 2026-08-07, before a1_btpeval_shallowcopy ran). Refusing costs a build; the alternative is a
# wrong answer that looks right.
if [ -n "${GOHEAP:-}" ]; then
    _cand="$BASE/sif/arion-${IMG_STEP}_goheap.sif"
    case "$IMG_STEP" in
        a0_default|p_*) _cand="$BASE/sif/arion-p_goheap.sif" ;;
    esac
    SIF="${SIF_GOHEAP:-$_cand}"
    [ -r "$SIF" ] || {
        echo "FAILED: no census image at $SIF; the Go adapter lives there, and so does the" >&2
        echo "        step's lever. Build one (arion-${IMG_STEP}_goheap.def) rather than letting" >&2
        echo "        this fall back to an image of a different program." >&2
        exit 2
    }
fi
[ -r "$SIF" ] || { echo "no image at $SIF" >&2; exit 2; }

RES="$BASE/results/$STEP${RUN:+/run$RUN}"
snni_prepare_results "$RES" || exit $?
mkdir -p "$RES/Output"

RUNNER="$BASE/run_arion.sh"
[ -r "$RUNNER" ] || { echo "no runner at $RUNNER" >&2; exit 2; }

# THE WORKLOAD: BERT-base at logN=16, rows=128, batch=256 (menu option 12), at a DECLARED depth of
# two transformer layers, not twelve: MOAI-CPU (its comparison point) is ~144 h/run at full depth,
# which this thesis cannot afford. Declared here, in RUN_META and the WORKLOAD| marker, and carried
# on every published row, rather than left implicit in which image ran (the earlier campaign's way).
export ARION_N_LAYERS="${ARION_N_LAYERS:-2}"
export ARION_THREADS="${ARION_THREADS:-64}"
export ARION_MENU="${ARION_MENU:-12}"
# GOGC at Go's default: the earlier campaign used the GC cap only as an OOM enabler, never a step,
# and this baseline needs none on a 1.5 TB node.
export GOGC="${GOGC:-100}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

# ONE RUN PER STEP (PROCEDURE.md, ONBOARDING.md "One run either way"). With SNNI_COMBINED=1 the Go
# heap census runs INSIDE the measured run: the poller calls census_hook.sh on material new maxima
# above SNNI_CENSUS_MIN_KB, the last capture is the one nearest the peak, and its pause is counted in
# frozen_ms. Without SNNI_COMBINED the old separate census channel below is used (retired).
COMBINED=""
if [ -n "${GOHEAP:-}" ] && [ -n "${SNNI_COMBINED:-}" ]; then
    COMBINED=1
    : "${SNNI_CENSUS_MIN_KB:?combined ARION run needs SNNI_CENSUS_MIN_KB (declared census floor)}"
    export SNNI_CENSUS_PCT="${SNNI_CENSUS_PCT:-5}"
    [ -r "$BASE/census_hook.sh" ] || { echo "FAILED: no $BASE/census_hook.sh" >&2; exit 2; }
fi

snni_run_meta "$RES" \
    "image=$SIF" \
    "image_digest=$(sha256sum "$SIF" | awk '{print $1}')" \
    "arion_n_layers=$ARION_N_LAYERS" \
    "arion_threads=$ARION_THREADS" \
    "arion_menu=$ARION_MENU" \
    "gogc=$GOGC" \
    "vram=NA (CPU system, single pool -- the degenerate case of the two-pool protocol)" \
    "model=bert-base logN=16 rows=128 batch=256 layers=$ARION_N_LAYERS (DECLARED SCOPE)" \
    "protocol=RNS-CKKS non-interactive FHE, single process, no parties" \
    "role=the single process; there is no second party to pair a decomposition with" \
    "combined_channel=${COMBINED:-0}" \
    "census=$([ -n "$COMBINED" ] && echo "in-run go heap dump, min_kb=$SNNI_CENSUS_MIN_KB pct=$SNNI_CENSUS_PCT" || echo "${GOHEAP:+separate channel}")"

T0=$(date +%s)

INNER="$(snni_pmrec_inner_env "$RES") export ARION_RESULTS=$RES"
INNER="$INNER ARION_N_LAYERS=$ARION_N_LAYERS ARION_THREADS=$ARION_THREADS ARION_MENU=$ARION_MENU"
INNER="$INNER GOGC=$GOGC"
# The adapter arms only if SNNI_HEAPDUMP_PATH is set; SNNI_PM_MIN sets MemProfileRate to the same
# campaign floor the malloc recorder uses. Both were missing from the first two attempts
# (45835492, 45836947), correctly refused by the guard.
if [ -n "$COMBINED" ]; then
    mkdir -p "$RES/census"
    INNER="$INNER SNNI_HEAPDUMP_PATH=$RES/census/heap.dump.new SNNI_PM_MIN=$SNNI_PM_MIN"
elif [ -n "${GOHEAP:-}" ]; then
    INNER="$INNER SNNI_HEAPDUMP_PATH=$RES/heap.dump SNNI_PM_MIN=$SNNI_PM_MIN"
fi
INNER="$INNER; exec bash $RUNNER"

apptainer exec \
    --bind "$RES":"$RES" \
    --bind "$RUNNER":"$RUNNER" \
    --bind "$RES/Output":/root/arion/Output \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

# run_arion.sh writes the pid of the Go process it launched; matched via the file per campaign rule.
PIDS=""
for _ in $(seq 1 3000); do
    PIDS=$(snni_worker_pids_from_files "$RES" 'arion.pid')
    [ -n "$PIDS" ] && kill -0 "$PIDS" 2>/dev/null && break
    kill -0 "$APP_PID" 2>/dev/null || break
    sleep 0.1
done

if [ -z "$PIDS" ]; then
    echo "ERROR: no payload pid appeared; this run has no peak" >&2
    RC=4
    wait "$APP_PID" 2>/dev/null
else
    echo "payload: $PIDS" | tee -a "$RES/poller_attach.log"
if [ -n "${GOHEAP:-}" ] && [ -z "$COMBINED" ]; then
    # =========================================================================================
    # THE GO HEAP CHANNEL: Go's runtime bypasses malloc, so the LD_PRELOAD recorder sees nothing
    # (baseline decomposed 100% `anon:residual`, 600,485,860 kB unnamed) and this channel supplies
    # the table pmrec gives other systems. Run as channel N of the SAME step (not a separate probe
    # step, which K1/K2 forbid pairing with the published peak), mirroring `_pmrec`/`_attrib`.
    # Mechanism lifted unchanged from `p_goheap.sbatch`.
    grep -qs "SNNI_HEAPDUMP|armed" "$RES/markers.txt" "$RES/stderr.log" || {
        sleep 5
        grep -qs "SNNI_HEAPDUMP|armed" "$RES/markers.txt" "$RES/stderr.log" || {
            echo "FAILED: the adapter did not announce itself; this binary would ignore the" >&2
            echo "        signal, so the channel would produce an empty table and look like a" >&2
            echo "        quiet system rather than a missing mechanism." >&2
            kill "$APP_PID" 2>/dev/null
            RC=5
        }
    }
    TRIGGERED=0
    while kill -0 "$PIDS" 2>/dev/null; do
        RSS=$(awk '/VmRSS/{print $2}' "/proc/$PIDS/status" 2>/dev/null || echo 0)
        if [ "${RSS:-0}" -ge "$TRIGGER_KB" ] && [ "$TRIGGERED" = 0 ]; then
            TRIGGERED=1
            echo "TRIGGER at rss_kb=$RSS ($(date -u +%H:%M:%S))" | tee -a "$RES/census.log"
            # FREEZE, THEN MEASURE -- the campaign's own rule, applied by hand here because the
            # snapshot and the dump are two operations that must describe one instant.
            kill -STOP "$PIDS"
            for _ in $(seq 1 100); do
                st=$(awk '{print $3}' "/proc/$PIDS/stat" 2>/dev/null || echo ?)
                [ "$st" = "T" ] && break
                sleep 0.05
            done
            awk '/VmRSS/{print $2}' "/proc/$PIDS/status" > "$RES/rss_at_snapshot_kb.txt"
            awk '/VmHWM/{print $2}' "/proc/$PIDS/status" > "$RES/hwm_at_snapshot_kb.txt"
            cp "/proc/$PIDS/smaps" "$RES/snapshot.smaps" 2>/dev/null || true
            # THE SNAPSHOT IS CHECKED BEFORE THE DUMP IS PAID FOR: measured 2026-08-06, a run spent
            # an hour and 570 GB on a dump with no pagemap.bin, so join_residency.py had nothing to
            # join it to, and the failure was invisible because stderr wasn't merged into the log.
            # Both are now fixed: stderr merged, and the dump is requested only after a non-empty
            # snapshot.
            python3 "$CAMP/attrib_go/pagemap_snapshot.py" "$PIDS" "$RES/pagemap.bin" 2>&1 \
                | tee -a "$RES/census.log"
            if [ ! -s "$RES/pagemap.bin" ]; then
                echo "FAILED: no pagemap snapshot at $RES/pagemap.bin, so the dump could not be" >&2
                echo "        joined to residency. Not requesting it: the dump costs ~570 GB and" >&2
                echo "        an hour, and without residency it is unusable. See census.log for" >&2
                echo "        the snapshot's own reason." >&2
                kill -CONT "$PIDS" 2>/dev/null
                kill -TERM "$PIDS" 2>/dev/null
                RC=8
                break
            fi
            kill -CONT "$PIDS"
            awk '/VmRSS/{print $2}' "/proc/$PIDS/status" > "$RES/rss_at_dump_kb.txt"
            kill -USR1 "$PIDS"
            echo "USR1 sent" | tee -a "$RES/census.log"
        fi
        sleep 5
    done
    RC=0; wait "$APP_PID" || RC=$?
    if [ "$TRIGGERED" != 1 ]; then
        echo "FAILED: rss never reached TRIGGER_KB=$TRIGGER_KB, so no census was taken" >&2
        [ "$RC" = 0 ] && RC=6
    else
        DUMP=$(ls "$RES"/heap.dump.* 2>/dev/null | head -1)
        if [ -n "$DUMP" ]; then
            python3 "$CAMP/attrib_go/parse_heapdump.py" "$DUMP" "$RES/objects_go.jsonl" \
                | tee -a "$RES/census.log" \
            && python3 "$CAMP/attrib_go/join_residency.py" "$RES/objects_go.jsonl" \
                "$RES/pagemap.bin" "$RES/objects_goheap.jsonl" \
                "$(cat "$RES/rss_at_snapshot_kb.txt")" | tee -a "$RES/census.log" \
            || { echo "FAILED: the dump was written but could not be turned into a table" >&2
                 [ "$RC" = 0 ] && RC=7; }
            sha256sum "$DUMP" > "$RES/heapdump.sha256"
            rm -f "$RES"/heap.dump.*
            gzip -f "$RES/objects_go.jsonl" 2>/dev/null || true
        else
            echo "FAILED: triggered but no heap dump was written" >&2
            [ "$RC" = 0 ] && RC=6
        fi
    fi
else
    if [ -n "$COMBINED" ]; then
        export CAMP SNNI_CENSUS_MIN_KB SNNI_CENSUS_PCT
        export SNNI_CENSUS_HOOK="$BASE/census_hook.sh" SNNI_CENSUS_DIR="$RES/census"
        export SNNI_HEAPDUMP_PATH="$RES/census/heap.dump.new" SNNI_CENSUS_STDERR="$RES/markers.txt"
        grep -qs "SNNI_HEAPDUMP|armed" "$RES/markers.txt" || sleep 5
        grep -qs "SNNI_HEAPDUMP|armed" "$RES/markers.txt" || {
            echo "FAILED: the Go adapter did not arm; this image cannot take the census" >&2
            kill "$APP_PID" 2>/dev/null; RC=5; }
    fi
    # shellcheck disable=SC2086
    POLL_PIDS=$(snni_start_poller "$RES" poll $PIDS)
    RC=${RC:-0}; wait "$APP_PID" || RC=$?
    # shellcheck disable=SC2086
    [ -n "$POLL_PIDS" ] && wait $POLL_PIDS 2>/dev/null
    # The run's wall ends HERE, when the program and its poller are done. Parsing the dump below is
    # post-processing; shifting T0 by its duration keeps it out of wall_seconds.txt.
    T_PROG_END=$(date +%s)
    if [ -n "$COMBINED" ]; then
        CD="$RES/census"
        if [ -s "$CD/heap.dump" ] && [ -s "$CD/pagemap.bin" ]; then
            echo "census taken at $(cat "$CD/CENSUS_AT") (ts rss_kb)" | tee -a "$CD/census.log"
            python3 "$CAMP/attrib_go/parse_heapdump.py" "$CD/heap.dump" "$CD/objects_go.jsonl" \
                | tee -a "$CD/census.log" \
            && python3 "$CAMP/attrib_go/join_residency.py" "$CD/objects_go.jsonl" \
                "$CD/pagemap.bin" "$RES/objects_goheap.jsonl" \
                "$(cat "$CD/rss_at_snapshot_kb.txt")" | tee -a "$CD/census.log" \
            || { echo "FAILED: the in-run dump could not be turned into a table" >&2
                 [ "$RC" = 0 ] && RC=7; }
            sha256sum "$CD/heap.dump" > "$CD/heapdump.sha256"
            rm -f "$CD/heap.dump" "$CD"/heap.dump.new.*
            gzip -f "$CD/objects_go.jsonl" 2>/dev/null || true
        else
            echo "FAILED: no in-run census was captured (peak below SNNI_CENSUS_MIN_KB?)" >&2
            [ "$RC" = 0 ] && RC=6
        fi
    fi
fi
fi

# The census stops the process mid-run, so it may not reach the gate output; not required here.
if { [ -z "${GOHEAP:-}" ] || [ -n "$COMBINED" ]; } && [ ! -s "$RES/gate.txt" ]; then
    echo "FAILED: no gate observable in $RES/gate.txt" >&2
    [ "$RC" = 0 ] && RC=5
fi

[ -n "${T_PROG_END:-}" ] && T0=$(( T0 + $(date +%s) - T_PROG_END ))
snni_finish "$RES" "$T0" "$RC"
exit $RC
