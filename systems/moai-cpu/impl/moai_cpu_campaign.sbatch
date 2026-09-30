#!/bin/bash
#SBATCH --job-name=moaicpu
# Partition zen4 since 2026-08-19 (serial chain: ~2 h queue vs ~6 days on bigsmp); baseline
# re-measured, MOAI/ARION bigsmp pairing lost.
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
# 24 h since 2026-08-28 (m2_pool_threshold projected 15h26 > the old 15 h); zen4 maximum 2 days.
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/slurm-%x-%j.err
#
# MOAI-CPU (SEAL 4.1 + bootstrapping fork, pure FHE) on the shared campaign harness.
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
AR=$CAMP/attrib_resident
BASE=$CAMP/moai-cpu

. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy

# Depth 7, not 5: at shallower depths the top rows are the allocator itself or generic Ciphertext
# copy/resize (not an object); depth 7 reaches program lines. Recorded in RUN_META since tables at
# different depths aren't comparable. Currently clamped to an effective 6 by pmrec's cap; left at 7
# (rather than lowered to 6) so it stays comparable with `m1_minimal_att_keys`, already run at 6.
export SNNI_PM_DEPTH="${SNNI_PM_DEPTH:-7}"

STEP="${STEP:?need STEP}"
IMG_STEP="${STEP%_pmrec}"; IMG_STEP="${IMG_STEP%_attrib}"

# THE ATTRIB GUARD (2026-09-17). A measurement step submitted without PMREC has attrib_host=off
# (measure.sh) and writes NO object table -- only PEAK and gate. That silent omission cost
# phase_split_fixed and m8_fixed a ~30h run each: both were hand-written direct sbatch that carried
# SNNI_POOLREC but forgot PMREC=1, so the pmsample freeze never wrote resident.txt and the derive
# had nothing to resolve. `run_step.sh` sets PMREC from the conf CHANNELS; a direct sbatch must
# carry it itself. Probes may skip it. Fail at ~0s, not at 30h.
case "$STEP" in
    p_*|x_*) ;;
    *)  [ -n "${PMREC:-}" ] || {
            echo "FAILED: measurement step '$STEP' submitted without PMREC=1 -> attrib_host=off ->" >&2
            echo "        no object table (PEAK and gate only). Add PMREC=1 to the --export, or" >&2
            echo "        submit via harness/run_step.sh (which sets it from the conf CHANNELS)." >&2
            exit 2
        } ;;
esac

SIF="${SIF:-$BASE/sif/moai-cpu-$IMG_STEP.sif}"
# Fallback is for probes only, not steps: a waterfall step falling back would silently measure
# the baseline binary under its own name, so steps are refused instead.
if [ ! -r "$SIF" ]; then
    case "${STEP:-}" in
        p_*|x_*|*_default)
            SIF="$BASE/sif/moai-cpu-m0_default.sif" ;;
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

RUNNER="$BASE/run_moai_cpu.sh"
[ -r "$RUNNER" ] || { echo "no runner at $RUNNER" >&2; exit 2; }

# Workload: BERT-base, batch 256, 128 rows, DECLARED depth of 2 of 12 layers (matches ARION;
# 12 layers is >4 days/replicate). Upstream default is 12, so this is set explicitly, not left unset.
export MOAI_N_LAYERS="${MOAI_N_LAYERS:-2}"
# Pinned at 16, not inherited: thread count is a large lever here (peak fell 132.7->90.2->73.1 GiB
# over 72->32->16 threads). T=72 runs predate the 2026-08-03 unification; kept as off-path evidence.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export SNNI_SEED="${SNNI_SEED:-0x5EED}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

snni_run_meta "$RES" \
    "image=$SIF" \
    "image_digest=$(sha256sum "$SIF" | awk '{print $1}')" \
    "omp_num_threads=$OMP_NUM_THREADS" \
    "snni_seed=$SNNI_SEED" \
    "moai_n_layers=$MOAI_N_LAYERS" \
    "vram=NA (CPU system, single pool -- the degenerate case of the two-pool protocol)" \
    "model=bert-base num_X=256 num_row=128 num_col=768 layers=$MOAI_N_LAYERS of 12 (DECLARED SCOPE)" \
    "protocol=CKKS logN=16 logp=46 logq=51 remaining_level=20 boot_level=14 hamming_weight=192" \
    "role=the single process; there is no second party to pair a decomposition with"

T0=$(date +%s)

INNER="$(snni_pmrec_inner_env "$RES") export MOAI_RESULTS=$RES"
INNER="$INNER MOAI_N_LAYERS=$MOAI_N_LAYERS OMP_NUM_THREADS=$OMP_NUM_THREADS SNNI_SEED=$SNNI_SEED"
# Optional: ohne SNNI_POOLREC wird die Pool-Tabelle des p_prm-Images nicht durchgereicht.
INNER="$INNER${SNNI_POOLREC:+ SNNI_POOLREC=$SNNI_POOLREC SNNI_POOLREC_TABLE=$RES/pooltable_%d}"
INNER="$INNER; exec bash $RUNNER"

apptainer exec \
    --bind "$RES":"$RES" \
    --bind "$RUNNER":"$RUNNER" \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

PIDS=""
for _ in $(seq 1 3000); do
    PIDS=$(snni_worker_pids_from_files "$RES" 'moai.pid')
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

# The program's own VmHWM trace, beside the poller's: monotone and exact, no sampling risk.
if [ -s "$RES/all_markers.txt" ]; then
    echo "=== the program's own VmHWM trace"
    tail -20 "$RES/all_markers.txt"
fi

snni_finish "$RES" "$T0" "$RC"

# Scratch hygiene (2026-09-19): a run's work/ holds ~50 GB of phase keys plus any parked key/copy
# files that only this run reads. Ten concurrent runs pushed the user's scratch usage to 6 TB, the
# quota, and seven runs died at once with SEAL's "I/O error" (exit 134) on their next write. The
# measurement (markers, poll/, gate, tables) lives outside work/, so nothing published is touched.
rm -rf "$RES/work/phase_keys" "$RES"/work/key_*_parked.dat
exit $RC
