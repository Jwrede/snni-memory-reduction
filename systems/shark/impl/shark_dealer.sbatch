#!/bin/bash
#SBATCH --job-name=sharkdealer
#SBATCH --partition=zen2-128C-496G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/shark/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/shark/slurm-%x-%j.err
#
# SHARK on the shared campaign harness (from tools/palma/templates/system.sbatch; shared logic in
# ../harness/measure.sh and ../harness/campaign.env).
# Partition zen2-128C-496G: ~108 GB combined online-party memory; all steps here. --exclusive and
# pinned threads required.
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
AR=$CAMP/attrib_resident
BASE=$CAMP/shark

. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy

STEP="${STEP:?need STEP}"
IMG_STEP="${STEP%_pmrec}"; IMG_STEP="${IMG_STEP%_attrib}"
# Dealer line measures the same binary as online (one build, three parties; step name selects the
# role polled, not the image), so it falls back to the online baseline image rather than looking
# for shark-d0_default.sif, which has never existed.
SIF="${SIF:-$BASE/sif/shark-$IMG_STEP.sif}"
# Fallback is for probes/refuted-attempts/baseline only (p_*, x_*, *_default); a waterfall step
# must run its own image or risk measuring the wrong binary (happened 2026-08-05, s2_threads_4).
if [ ! -r "$SIF" ]; then
    case "${STEP:-}" in
        p_*|x_*|*_default)
            SIF="$BASE/sif/shark-s0_default.sif" ;;
        *)
            echo "FAILED: step '${STEP:-}' has no image at $SIF, and a waterfall step may not fall" >&2
            echo "        back to the baseline: it would measure the baseline binary under this" >&2
            echo "        step's name. Build its image, or name one explicitly through SIF." >&2
            exit 2 ;;
    esac
fi
[ -r "$SIF" ] || { echo "no image at $SIF" >&2; exit 2; }

RES="$BASE/results/$STEP${RUN:+/run$RUN}"
snni_prepare_results "$RES" || exit $?

# ~100 GB of preprocessing per run, on a path of this run's own so that two runs can never read
# each other's files. Removed by the inner script at the end and again here.
WORK=/scratch/tmp/j_wred02/shark_work/$STEP${RUN:+_run$RUN}
rm -rf "$WORK"; mkdir -p "$WORK" || exit 2

# Threads pinned to 16 (campaign decision 2026-08-03); a T4/T8/T16 sweep moved the peak <0.3% at
# most steps. Pinned rather than inherited so the peak isn't a property of the 128-core node.
export OMP_NUM_THREADS=16

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

# Library lineage hashed per run: patches don't accumulate across step builds, so a step missing
# its cumulative patch silently loses earlier levers (happened 2026-07-30, s5_up_split).
LIBFP=$(apptainer exec "$SIF" sh -c \
    'cat /root/shark/src/protocols/lrs.cpp /root/shark/include/shark/utils/comm.hpp 2>/dev/null | md5sum' \
    2>/dev/null | awk '{print $1}')

snni_run_meta "$RES" \
    "image=$SIF" \
    "image_digest=$(sha256sum "$SIF" | awk '{print $1}')" \
    "lib_fingerprint=${LIBFP:-UNKNOWN}" \
    "omp_num_threads=$OMP_NUM_THREADS" \
    "vram=NA (CPU system, single pool -- the degenerate case of the two-pool protocol)" \
    "model=bert-base n_embd=768 n_layer=12 n_seq=128 f=16 (SHARK_N_LAYERS=12 SHARK_SEQ_LEN=128)" \
    "role=THE DEALER (party 2). The online parties still run, for the gate, and are NOT polled here" \
    "freeze_leader_party=${SHARK_FREEZE_LEADER:-0}"

# T0_JOB is the whole job's clock (record only); T0 becomes wall_seconds.txt and resets when
# phase 2 begins, since the measured role is the runtime axis, not the dealer phase.
T0_JOB=$(date +%s)
T0=$T0_JOB

# Environment goes into an inner shell rather than apptainer --env: some site configs manage
# LD_PRELOAD themselves and would silently drop it, giving an unrecorded run with no error.
INNER="$(snni_pmrec_inner_env "$RES") export SHARK_RESULTS=$RES SHARK_WORK=$WORK"
INNER="$INNER SHARK_STEP=$STEP"
INNER="$INNER SHARK_N_LAYERS=12 SHARK_SEQ_LEN=128 OMP_NUM_THREADS=$OMP_NUM_THREADS"
INNER="$INNER; exec bash $BASE/run_shark_dealer.sh"

apptainer exec \
    --bind "$RES":"$RES" \
    --bind "$WORK":"$WORK" \
    --bind "$BASE/run_shark_dealer.sh":"$BASE/run_shark_dealer.sh" \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

# Waits for dealer.pid (phase 1's subject), not phase 2, since all three parties are the same
# binary. One pid, not two, matching that the dealer is a single process -- the inherited two-pid
# guard could never be satisfied and once failed a fine run (job 45766695). Ceiling is generous
# (~112 GB write).
PIDS=""
for _ in $(seq 1 36000); do
    PIDS=$(snni_worker_pids_from_files "$RES" 'dealer.pid')
    [ "$(echo "$PIDS" | wc -w)" -ge 1 ] && break
    kill -0 "$APP_PID" 2>/dev/null || break
    sleep 0.1
done

if [ "$(echo "$PIDS" | wc -w)" -lt 1 ]; then
    echo "ERROR: the dealer process never appeared; this run has no peak" >&2
    RC=4
    wait "$APP_PID" 2>/dev/null
else
    # Runtime axis is phase 1 here; online phase is recorded separately (online_wall_seconds.txt),
    # since a dealer lever could change what online parties have to read.
    T0=$(date +%s)
    echo "dealer: $PIDS" | tee -a "$RES/poller_attach.log"
    # Single process is trivially its own freeze leader (no per-step leader choice needed). Its
    # trace directory name is left to the poller (`poll`), not hand-written here: a hand-written
    # guess once named a nonexistent directory and killed six derives (45772764-66).
    # shellcheck disable=SC2086
    POLL_PIDS=$(snni_start_poller "$RES" poll $PIDS)
    RC=0; wait "$APP_PID" || RC=$?
    # shellcheck disable=SC2086
    [ -n "$POLL_PIDS" ] && wait $POLL_PIDS 2>/dev/null
fi

# The gate is produced by the inner script; the sbatch only refuses a run that has none. A run
# that produced no gate output is not a run with a passing gate.
if [ ! -s "$RES/gate.txt" ]; then
    echo "FAILED: no gate observable at $RES/gate.txt" >&2
    [ "$RC" = 0 ] && RC=5
fi

rm -rf "$WORK"
# Runtime axis is phase 1, set by adjusting T0 (not by overwriting wall_seconds.txt afterward,
# which would desync it from PEAK.txt): passing the real T0 would cover both phases (601s vs 307s
# dealer-only in run 45769252). Online phase stays visible separately in online_wall_seconds.txt.
_dealer_wall=$(cat "$RES/dealer_wall_seconds.txt" 2>/dev/null || echo "")
case "$_dealer_wall" in
    ''|*[!0-9]*) ;;                                   # unusable, keep the honest job-wide T0
    *) T0=$(( $(date +%s) - _dealer_wall )) ;;
esac
snni_finish "$RES" "$T0" "$RC"
{
    echo "job_wall_s=$(( $(date +%s) - T0_JOB ))"
    echo "dealer_wall_s=$(cat "$RES/dealer_wall_seconds.txt" 2>/dev/null || echo NA)"
    echo "online_wall_s=$(cat "$RES/online_wall_seconds.txt" 2>/dev/null || echo NA)"
    echo "note=wall_s above is the MEASURED phase 1 (the dealer); the online phase is recorded separately"
} >> "$RES/PEAK.txt"
tail -4 "$RES/PEAK.txt"
exit $RC
