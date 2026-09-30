#!/bin/bash
#SBATCH --job-name=shark
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
# SNNI_AR selects a recorder build other than the live one (opt-in, 2026-09-29): a line re-measured
# with the build it was first measured with, from $CAMP/recorder_builds/<build>/. RUN_META records
# it (snni_run_meta). Unset, the live attrib_resident as before.
AR="${SNNI_AR:-$CAMP/attrib_resident}"
BASE=$CAMP/shark

. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy

STEP="${STEP:?need STEP}"
IMG_STEP="${STEP%_pmrec}"; IMG_STEP="${IMG_STEP%_attrib}"

# p_alloc_* probe: pins glibc's allocator on an existing step's image (diagnostic, not a waterfall
# row); motivated by SHAFT-CPU's -12.48% result measuring this same knob.
case "$IMG_STEP" in
    p_alloc_*) IMG_STEP="${IMG_STEP#p_alloc_}"
               SNNI_PIN_ALLOC=1 ;;
    # p_seq<N>_* probe: runs an existing image at a different sequence length (a geometry
    # diagnostic, not a waterfall row).
    p_seq*_*)  SNNI_SEQ="${IMG_STEP#p_seq}"; SNNI_SEQ="${SNNI_SEQ%%_*}"
               IMG_STEP="${IMG_STEP#p_seq${SNNI_SEQ}_}"
               echo "GEOMETRY_PROBE=1 n_seq=$SNNI_SEQ on image $IMG_STEP" ;;
esac
SIF="${SIF:-$BASE/sif/shark-$IMG_STEP.sif}"
[ -r "$SIF" ] || { echo "no image at $SIF" >&2; exit 2; }

RES="$BASE/results/$STEP${RUN:+/run$RUN}"
snni_prepare_results "$RES" || exit $?

# ~100 GB of preprocessing per run, on a path of this run's own so that two runs can never read
# each other's files. Removed by the inner script at the end and again here.
WORK=/scratch/tmp/j_wred02/shark_work/$STEP${RUN:+_run$RUN}
rm -rf "$WORK"; mkdir -p "$WORK" || exit 2

# Threads pinned to 16, the campaign's unified count (2026-08-03); SHARK has no thread setting of its
# own. SHARK_OMP overrides it for an operating-point study only (the T=4 line is archived). Pinned
# rather than inherited so the peak is not a property of the 128-core node.
export OMP_NUM_THREADS=${SHARK_OMP:-16}

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
    "role=ONLINE parties 0 and 1 (server + client); the dealer runs first and is NOT measured" \
    "freeze_leader_party=${SHARK_FREEZE_LEADER:-0}"

# T0_JOB is the whole job's clock (record only); T0 becomes wall_seconds.txt and resets when
# phase 2 begins, since the measured role is the runtime axis, not the dealer phase.
T0_JOB=$(date +%s)
T0=$T0_JOB

# Environment goes into an inner shell rather than apptainer --env: some site configs manage
# LD_PRELOAD themselves and would silently drop it, giving an unrecorded run with no error.
INNER="$(snni_pmrec_inner_env "$RES") export SHARK_RESULTS=$RES SHARK_WORK=$WORK"
INNER="$INNER SHARK_STEP=$STEP"
INNER="$INNER SHARK_N_LAYERS=12 SHARK_SEQ_LEN=${SNNI_SEQ:-128} OMP_NUM_THREADS=$OMP_NUM_THREADS"
# Malloc tunables match SHAFT-CPU's verbatim so both systems' probes answer the same question;
# setting any of them disables glibc's dynamic mmap-threshold adaptation, the mechanism under test.
if [ "${SNNI_PIN_ALLOC:-0}" = "1" ]; then
    INNER="$INNER MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=0 MALLOC_MMAP_THRESHOLD_=131072"
    echo "ALLOC_PINNED=1 (MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=0 MALLOC_MMAP_THRESHOLD_=131072)"
fi
INNER="$INNER; exec bash $BASE/run_shark_measured.sh"

apptainer exec \
    --bind "$RES":"$RES" \
    --bind "$WORK":"$WORK" \
    --bind "$BASE/run_shark_measured.sh":"$BASE/run_shark_measured.sh" \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

# Waits for phase 2's pid files (party0/1.pid), not a process name, since all three parties are
# the same binary; this keeps key generation out of the measurement. Ceiling is generous (~100 GB write).
PIDS=""
for _ in $(seq 1 36000); do
    PIDS=$(snni_worker_pids_from_files "$RES" 'party*.pid')
    [ "$(echo "$PIDS" | wc -w)" -ge 2 ] && break
    kill -0 "$APP_PID" 2>/dev/null || break
    sleep 0.1
done

if [ "$(echo "$PIDS" | wc -w)" -lt 2 ]; then
    echo "ERROR: phase 2 never started (fewer than two online pids); this run has no peak" >&2
    RC=4
    wait "$APP_PID" 2>/dev/null
else
    # Runtime axis is the measured phase (online), not the whole job: including the ~307s dealer
    # phase would dilute runtime_delta_pct roughly 4x. Dealer wall is recorded separately, in
    # dealer_wall_seconds.txt.
    T0=$(date +%s)
    echo "online parties: $PIDS" | tee -a "$RES/poller_attach.log"
    # Freeze leader (first pid passed to snni_start_poller) must be the taller party, since only
    # its capture is atomic; which party is taller moved mid-line (party 0 through s3, party 1
    # from s4 onward) and is recorded per run in RUN_META rather than fixed as an instrument setting.
    LEADER_PARTY="${SHARK_FREEZE_LEADER:-0}"
    P0=$(tr -dc '0-9' < "$RES/party0.pid")
    P1=$(tr -dc '0-9' < "$RES/party1.pid")
    if [ "$LEADER_PARTY" = 1 ]; then ORDERED="$P1 $P0"; else ORDERED="$P0 $P1"; fi
    echo "freeze leader: party $LEADER_PARTY (pid ${ORDERED%% *})" | tee -a "$RES/poller_attach.log"
    # shellcheck disable=SC2086
    POLL_PIDS=$(snni_start_poller "$RES" poll $ORDERED)
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
snni_finish "$RES" "$T0" "$RC"
{
    echo "job_wall_s=$(( $(date +%s) - T0_JOB ))"
    echo "dealer_wall_s=$(cat "$RES/dealer_wall_seconds.txt" 2>/dev/null || echo NA)"
    echo "online_wall_s=$(cat "$RES/online_wall_seconds.txt" 2>/dev/null || echo NA)"
    echo "note=wall_s above is the MEASURED phase 2; the dealer phase is recorded separately"
} >> "$RES/PEAK.txt"
tail -4 "$RES/PEAK.txt"
exit $RC
