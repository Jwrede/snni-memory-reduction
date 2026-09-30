#!/bin/bash
#SBATCH --job-name=bb
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/bumblebee/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/bumblebee/slurm-%x-%j.err
#
# BumbleBee (Cheetah 2PC over SPU) on the shared campaign harness (from tools/palma/templates/system.sbatch).
# Partition express for every step (same node type as normal). --exclusive and pinned threads
# required: each OT instance holds two 160 MiB Ferret buffers (yacl_ote_adapter.h:58,114).
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
AR=$CAMP/attrib_resident
BASE=$CAMP/bumblebee

. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy

STEP="${STEP:?need STEP}"
# Channel suffixes are stripped to find the image: <step>_pmrec and <step>_attrib measure the same
# step through different instruments and must run the SAME binary.
IMG_STEP="${STEP%_pmrec}"; IMG_STEP="${IMG_STEP%_attrib}"
# A `p_seq<N>_*` probe runs an existing step's image at a different sequence length (builds
# nothing new); 197 = ViT-Base/16's patch count (196+CLS) on BERT-base's transformer shapes. A
# diagnostic only, never a waterfall row (different operating points aren't comparable).
case "$IMG_STEP" in
    p_seq*_*) SNNI_SEQ="${IMG_STEP#p_seq}"; SNNI_SEQ="${SNNI_SEQ%%_*}"
              IMG_STEP="${IMG_STEP#p_seq${SNNI_SEQ}_}"
              echo "GEOMETRY_PROBE=1 n_seq=$SNNI_SEQ on image $IMG_STEP" ;;
esac


# Steps b5-b8 have no image of their own by design: they're b4's binary at a different OT-instance
# count (env value), so the mapping is a table here rather than left to whoever submits the job.
case "$IMG_STEP" in
    b5_ot_instances_8) IMG_STEP=b4_slice_budget_64m; BB_MAX_CONCURRENCY="${BB_MAX_CONCURRENCY:-8}" ;;
    b6_ot_instances_4) IMG_STEP=b4_slice_budget_64m; BB_MAX_CONCURRENCY="${BB_MAX_CONCURRENCY:-4}" ;;
    b7_ot_instances_2) IMG_STEP=b4_slice_budget_64m; BB_MAX_CONCURRENCY="${BB_MAX_CONCURRENCY:-2}" ;;
    b8_ot_instances_1) IMG_STEP=b4_slice_budget_64m; BB_MAX_CONCURRENCY="${BB_MAX_CONCURRENCY:-1}" ;;
esac
SIF="${SIF:-$BASE/sif/bumblebee-$IMG_STEP.sif}"
[ -r "$SIF" ] || { echo "no image at $SIF" >&2; exit 2; }

RES="$BASE/results/$STEP${RUN:+/run$RUN}"
snni_prepare_results "$RES" || exit $?

# Workload: BERT-base, 12 layers, seq len 128, identical for every published step; overridable
# only so a diagnostic can vary shape (never a waterfall step at another value). RUN_META records what was used.
export BB_N_LAYERS="${BB_N_LAYERS:-12}"
export BB_SEQ_LEN="${BB_SEQ_LEN:-${SNNI_SEQ:-128}}"
# Pinned (16/party, not inherited) since it's part of the measured program (multiplies Ferret OT
# buffers); overridable only for the off-path diagnostic comparing against the retired 1-thread line.
export BB_MAX_CONCURRENCY="${BB_MAX_CONCURRENCY:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

# Recorder depth 5, derived from this system's own tables (impl/p_depth_choice.md), not borrowed
# from BOLT: it's the depth where the Ferret OT row separates into the instances b5-b8 actually
# halve. Set before snni_run_meta so RUN_META records pm_depth=5 (depths aren't comparable row for row).
export SNNI_PM_DEPTH="${SNNI_PM_DEPTH:-5}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

snni_run_meta "$RES" \
    "image=$SIF" \
    "image_digest=$(sha256sum "$SIF" | awk '{print $1}')" \
    "omp_num_threads=$OMP_NUM_THREADS" \
    "bb_max_concurrency=$BB_MAX_CONCURRENCY" \
    "vram=NA (CPU system, single pool -- the degenerate case of the two-pool protocol)" \
    "model=bert-base n_layer=$BB_N_LAYERS n_seq=$BB_SEQ_LEN hidden=768 heads=12 ffn=3072" \
    "protocol=CHEETAH field=FM64 fxp=16 ot=YACL_Ferret mul_lsb_error=true" \
    "role=BOTH parties (rank 0 and rank 1), measured together on one node"

T0=$(date +%s)

# Env goes into an inner shell (not `apptainer --env`): apptainer manages LD_PRELOAD itself on
# some sites, and a silently dropped preload gives an attribution run with no recorder and no error.
INNER="$(snni_pmrec_inner_env "$RES") export BB_RESULTS=$RES"
INNER="$INNER BB_N_LAYERS=$BB_N_LAYERS BB_SEQ_LEN=$BB_SEQ_LEN"
INNER="$INNER BB_MAX_CONCURRENCY=$BB_MAX_CONCURRENCY OMP_NUM_THREADS=$OMP_NUM_THREADS"
# Link receive timeout, forwarded only when set (a failure detector, not protocol); needed once
# when a single OT worker at 197 tokens left gaps longer than yacl's default (65 s timeout x3).
[ -n "${BB_RECV_TIMEOUT_MS:-}" ] && INNER="$INNER BB_RECV_TIMEOUT_MS=$BB_RECV_TIMEOUT_MS"
INNER="$INNER; exec bash $BASE/run_bumblebee.sh"

apptainer exec \
    --bind "$RES":"$RES" \
    --bind "$BASE/run_bumblebee.sh":"$BASE/run_bumblebee.sh" \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

# Ask the harness, don't pattern-match: both parties are the same binary, distinguishable only by
# the pid files run_bumblebee.sh writes.
PIDS=""
for _ in $(seq 1 3000); do
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

# The gate: refuses to pass a run with no gate.txt (written/checked by run_bumblebee.sh).
if [ ! -s "$RES/gate.txt" ]; then
    echo "FAILED: no gate observable in $RES/gate.txt" >&2
    [ "$RC" = 0 ] && RC=5
fi

snni_finish "$RES" "$T0" "$RC"
exit $RC
