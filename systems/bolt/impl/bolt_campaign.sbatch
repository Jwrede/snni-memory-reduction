#!/bin/bash
#SBATCH --job-name=bolt
#SBATCH --partition=bigsmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
# Time limit from the measured worst-case wall (1515 s); keeps bigsmp backfill (3 nodes) usable.
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/bolt/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/bolt/slurm-%x-%j.err
#
# BOLT (EzPC/SCI + SEAL 4.1.1, 2PC hybrid HE+MPC) on the shared campaign harness.
# ALICE alone 112.6 to 115.4 GiB at baseline. --exclusive and pinned threads are part of the
# measured configuration (as are the compile-time thread constants).
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
AR=$CAMP/attrib_resident
BASE=$CAMP/bolt

. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy

# Object table depth 5 names the program STAGE (not just the allocator/container/fill-line at
# depths 1/3/4), since this system allocates through SEAL's shared memory pool. Set here so every
# run carries it consistently; recorded in RUN_META since tables at different depths aren't comparable.
export SNNI_PM_DEPTH="${SNNI_PM_DEPTH:-5}"

STEP="${STEP:?need STEP}"
IMG_STEP="${STEP%_pmrec}"; IMG_STEP="${IMG_STEP%_attrib}"

SIF="${SIF:-$BASE/sif/bolt-$IMG_STEP.sif}"
# Fallback to the baseline image is for probes/refuted attempts (p_*, x_*, *_default) only: a
# waterfall step must run its own cumulative binary or it silently measures the wrong program.
if [ ! -r "$SIF" ]; then
    case "${STEP:-}" in
        p_*|x_*|*_default)
            SIF="$BASE/sif/bolt-s0_default.sif" ;;
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

RUNNER="$BASE/run_bolt.sh"
[ -r "$RUNNER" ] || { echo "no runner at $RUNNER" >&2; exit 2; }

# Workload: BERT-base, 12 layers, seq len 128, one SST-2 sample, identical for every published step.
export BOLT_ID="${BOLT_ID:-0}"
export BOLT_NUM_CLASS="${BOLT_NUM_CLASS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export SNNI_SEED="${SNNI_SEED:-0x5EED}"
# Port derived from job id so a stray listener from a previous run isn't mistaken for this one's peer.
export BOLT_PORT="${BOLT_PORT:-$(( 8000 + ${SLURM_JOB_ID:-0} % 1000 ))}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

snni_run_meta "$RES" \
    "image=$SIF" \
    "image_digest=$(sha256sum "$SIF" | awk '{print $1}')" \
    "omp_num_threads=$OMP_NUM_THREADS" \
    "snni_seed=$SNNI_SEED" \
    "bolt_sample_id=$BOLT_ID" \
    "bolt_port=$BOLT_PORT" \
    "vram=NA (CPU system, single pool -- the degenerate case of the two-pool protocol)" \
    "model=bert-base seq=128 layers=12 num_class=$BOLT_NUM_CLASS (SST-2 quantised weights)" \
    "protocol=BOLT 2PC, HE backend SEAL 4.1.1 (BFV), OT via SCI/KKOT, bitlength 37, NL_SCALE 12" \
    "role=BOTH parties (ALICE r=1 weights, BOB r=2 input), measured together on one node"

T0=$(date +%s)

INNER="$(snni_pmrec_inner_env "$RES") export BOLT_RESULTS=$RES"
INNER="$INNER BOLT_ID=$BOLT_ID BOLT_NUM_CLASS=$BOLT_NUM_CLASS BOLT_PORT=$BOLT_PORT"
INNER="$INNER OMP_NUM_THREADS=$OMP_NUM_THREADS SNNI_SEED=$SNNI_SEED"
# Inner shell is built by name, not inherited; MALLOC_ARENA_MAX passed only when set, so baseline keeps glibc's default.
INNER="$INNER${MALLOC_ARENA_MAX:+ MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX}"
# Same rule for the layer window: passed only when set, so the image defaults to k=1 (s1's behaviour) unless overridden.
INNER="$INNER${BOLT_LAYER_WINDOW:+ BOLT_LAYER_WINDOW=$BOLT_LAYER_WINDOW}"
INNER="$INNER; exec bash $RUNNER"

apptainer exec \
    --bind "$RES":"$RES" \
    --bind "$RUNNER":"$RUNNER" \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

# Ask the harness rather than pattern-match: both parties are the same binary with the same name.
PIDS=""
for _ in $(seq 1 6000); do
    PIDS=$(snni_worker_pids_from_files "$RES" 'party*.pid')
    [ "$(echo "$PIDS" | wc -w)" -ge 2 ] && break
    kill -0 "$APP_PID" 2>/dev/null || break
    sleep 0.1
done

if [ "$(echo "$PIDS" | wc -w)" -lt 2 ]; then
    echo "ERROR: fewer than two party pids appeared; this run has no peak" >&2
    RC=4
    wait "$APP_PID" 2>/dev/null
else
    echo "parties: $PIDS" | tee -a "$RES/poller_attach.log"
    # One leader freezes BOTH parties: stopping only one peer of a 2PC protocol is a protocol fault.
    # shellcheck disable=SC2086
    POLL_PIDS=$(snni_start_poller "$RES" poll $PIDS)
    RC=0; wait "$APP_PID" || RC=$?
    # shellcheck disable=SC2086
    [ -n "$POLL_PIDS" ] && wait $POLL_PIDS 2>/dev/null
fi

if [ ! -s "$RES/gate.txt" ]; then
    echo "FAILED: no gate observable in $RES/gate.txt" >&2
    [ "$RC" = 0 ] && RC=5
fi

snni_finish "$RES" "$T0" "$RC"
exit $RC
