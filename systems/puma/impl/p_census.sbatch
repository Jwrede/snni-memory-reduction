#!/bin/bash
#SBATCH --job-name=puma-census
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/puma/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/puma/slurm-%x-%j.err
#
# PUMA (SecretFlow SPU, ABY3 3PC) on the shared campaign harness (from tools/palma/templates/system.sbatch).
# Partition express for every step (36-core Skylake AVX-512, 95 GB; workload ~19 GiB). If a run
# exceeds 2 h, the whole line moves to normal. --exclusive required.
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
AR=$CAMP/attrib_resident
BASE=$CAMP/puma

. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy

STEP="${STEP:?need STEP}"
IMG_STEP="${STEP%_pmrec}"; IMG_STEP="${IMG_STEP%_attrib}"

SIF="${SIF:-$BASE/sif/puma-$IMG_STEP.sif}"
[ -r "$SIF" ] || { echo "no image at $SIF" >&2; exit 2; }

RES="$BASE/results/$STEP${RUN:+/run$RUN}"
snni_prepare_results "$RES" || exit $?

# The measured program is a bind-mounted script (md5 recorded per run by run_puma_census.sh);
# the image holds only the stack, so a step's lever can be a different driver/module, no rebuild.
DRIVER="${PUMA_DRIVER:-$BASE/snni_bert_puma.py}"
RUNNER="$BASE/run_puma_census.sh"
[ -r "$DRIVER" ] || { echo "no driver at $DRIVER" >&2; exit 2; }
[ -r "$RUNNER" ] || { echo "no runner at $RUNNER" >&2; exit 2; }

# Workload: BERT-base, 12 layers, seq len 128, identical for every published step; overridable
# only so a diagnostic can vary depth (never a waterfall step at another value). RUN_META records what was used.
export PUMA_N_LAYERS="${PUMA_N_LAYERS:-12}"
export PUMA_SEQ_LEN="${PUMA_SEQ_LEN:-128}"
# Pinned, not inherited: OMP doesn't bind SPU's own thread pool (not a real lever), but leaving it to the node would make the peak machine-dependent.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SNNI_SEED="${SNNI_SEED:-0x5EED}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

snni_run_meta "$RES" \
    "image=$SIF" \
    "image_digest=$(sha256sum "$SIF" | awk '{print $1}')" \
    "driver=$DRIVER" \
    "driver_md5=$(md5sum "$DRIVER" | awk '{print $1}')" \
    "omp_num_threads=$OMP_NUM_THREADS" \
    "snni_seed=$SNNI_SEED" \
    "vram=NA (CPU system, single pool -- the degenerate case of the two-pool protocol)" \
    "model=bert-base-uncased n_layer=$PUMA_N_LAYERS n_seq=$PUMA_SEQ_LEN hidden=768 heads=12" \
    "protocol=ABY3 field=FM64 3PC localhost cluster (3 SPU nodes + 2 PYU nodes + driver)" \
    "role=ALL six processes polled; the binding one is published"

T0=$(date +%s)

INNER="$(snni_pmrec_inner_env "$RES") export PUMA_RESULTS=$RES PUMA_DRIVER=$DRIVER"
INNER="$INNER PUMA_N_LAYERS=$PUMA_N_LAYERS PUMA_SEQ_LEN=$PUMA_SEQ_LEN"
INNER="$INNER OMP_NUM_THREADS=$OMP_NUM_THREADS SNNI_SEED=$SNNI_SEED"
INNER="$INNER; exec bash $RUNNER"

apptainer exec \
    --bind "$RES":"$RES" \
    --bind "$RUNNER":"$RUNNER" \
    --bind "$DRIVER":"$DRIVER" \
    --bind "$BASE/census":/opt/census \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

# Ask the harness, don't pattern-match: the five node services are indistinguishable by argv.
# The driver is polled too since it holds the plaintext model handed to P2.
PIDS=""
for _ in $(seq 1 6000); do
    PIDS=$(snni_worker_pids_from_files "$RES" 'node*.pid')
    DRVPID=$(snni_worker_pids_from_files "$RES" 'driver.pid')
    [ -n "$DRVPID" ] && PIDS="$PIDS $DRVPID"
    [ "$(echo "$PIDS" | wc -w)" -ge 6 ] && break
    kill -0 "$APP_PID" 2>/dev/null || break
    sleep 0.1
done

if [ "$(echo "$PIDS" | wc -w)" -lt 5 ]; then
    echo "ERROR: fewer than five cluster pids appeared; this run has no peak" >&2
    RC=4
    wait "$APP_PID" 2>/dev/null
else
    echo "cluster: $PIDS" | tee -a "$RES/poller_attach.log"
    # One leader freezes every process with one kill: stopping only one 3PC peer is a protocol fault.
    # shellcheck disable=SC2086
    POLL_PIDS=$(snni_start_poller "$RES" poll $PIDS)
    RC=0; wait "$APP_PID" || RC=$?
    # shellcheck disable=SC2086
    [ -n "$POLL_PIDS" ] && wait $POLL_PIDS 2>/dev/null
fi

# The gate: refuses to pass a run with no gate.txt (written/checked by run_puma_census.sh).
if [ ! -s "$RES/gate.txt" ]; then
    echo "FAILED: no gate observable in $RES/gate.txt" >&2
    [ "$RC" = 0 ] && RC=5
fi

snni_finish "$RES" "$T0" "$RC"
exit $RC
