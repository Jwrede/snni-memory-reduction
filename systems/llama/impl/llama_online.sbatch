#!/bin/bash
#SBATCH --job-name=llonline
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/llama/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/llama/slurm-%x-%j.err
#
# LLAMA online on the shared campaign harness (from tools/palma/templates/system.sbatch).
# Same machine type as the dealer line (express, 36-core Skylake, 95 GB, exclusive).
# --exclusive and a pinned thread count are required.
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
# SNNI_AR selects a recorder build other than the live one (opt-in, 2026-09-29): a line re-measured
# with the build it was first measured with, from $CAMP/recorder_builds/<build>/. RUN_META records
# it (snni_run_meta). Unset, the live attrib_resident as before.
AR="${SNNI_AR:-$CAMP/attrib_resident}"
BASE=$CAMP/llama

. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy

STEP="${STEP:?need STEP}"
IMG_STEP="${STEP%_pmrec}"; IMG_STEP="${IMG_STEP%_attrib}"
SIF="${SIF:-$BASE/sif/llama-online-$IMG_STEP.sif}"
# A diagnostic step has no image of its own, so probes (`p_*`, `x_*`) and the baseline fall back to
# the baseline image. A waterfall step must NOT fall back: its peak is cumulative against its
# predecessor, so falling back would measure the wrong binary under its name (as happened
# 2026-08-05: `bolt s2_threads_4` ran the baseline binary and reported 96.8 GB against s1's 17.5 GB).
if [ ! -r "$SIF" ]; then
    case "${STEP:-}" in
        p_*|x_*|*_default)
            SIF="$BASE/sif/llama-online-o0_default.sif" ;;
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

# 47 GB of key material per run; kept on the node's own scratch, not shared with other steps.
WORK=/scratch/tmp/j_wred02/llama_online_work/$STEP${RUN:+_run$RUN}
rm -rf "$WORK"; mkdir -p "$WORK" || exit 2

# Pinned to 16, same as the dealer line, so the two roles remain comparable.
export OMP_NUM_THREADS=16
export MALLOC_ARENA_MAX=2

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

snni_run_meta "$RES" \
    "image=$SIF" \
    "image_digest=$(sha256sum "$SIF" | awk '{print $1}')" \
    "llama_n_layers=${LLAMA_N_LAYERS:-12}" \
    "diag_no_freeze=${SNNI_DIAG_NO_FREEZE:-0}" \
    "omp_num_threads=$OMP_NUM_THREADS" \
    "malloc_arena_max=$MALLOC_ARENA_MAX" \
    "vram=NA (CPU system, single pool -- the degenerate case of the two-pool protocol)" \
    "model=bert-base n_embd=768 n_layer=12 bw=51 n_seq=128 scale=12" \
    "role=ONLINE parties 2 and 3 (server + client); the dealer runs first and is NOT measured"

T0=$(date +%s)

INNER="$(snni_pmrec_inner_env "$RES") export LLAMA_RESULTS=$RES LLAMA_WORK=$WORK"
INNER="$INNER OMP_NUM_THREADS=$OMP_NUM_THREADS MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"
# Declared runtime parameter defaulting to upstream's twelve layers, so an unset variable can never
# quietly shorten a measured run; the baseline image ignores it, so passing it costs nothing.
INNER="$INNER LLAMA_N_LAYERS=${LLAMA_N_LAYERS:-12}"

# DIAGNOSTIC ONLY. `poll_peak.sh` tests the freeze flag by PRESENCE, so `SNNI_PM_FREEZE=0` still
# reads as ON (a control run believed it disabled the freeze and measured the same config twice,
# 2026-08-04); the only honest way to disable it is to unset it entirely. Never set on a measured run.
if [ -n "${SNNI_DIAG_NO_FREEZE:-}" ]; then
    echo "DIAGNOSTIC: freeze disabled for this run (SNNI_DIAG_NO_FREEZE)"
    unset SNNI_PM_FREEZE
fi
INNER="$INNER; exec bash $BASE/run_llama_online.sh"

apptainer exec \
    --bind "$RES":"$RES" \
    --bind "$WORK":"$WORK" \
    --bind "$BASE/run_llama_online.sh":"$BASE/run_llama_online.sh" \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

# Wait for phase 2 via party2.pid/party3.pid (not process name, since all three parties share a
# binary): this keeps key generation out of the measurement. Ceiling is generous since phase 1 is
# a full dealer run (225-265 s published, more under a slow filesystem).
PIDS=""
for _ in $(seq 1 9000); do
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
    echo "online parties: $PIDS" | tee -a "$RES/poller_attach.log"
    # One leader freezes BOTH online parties with a single kill (stopping only one peer would be a
    # protocol fault, not a pause). The dealer already finished and is not in the freeze set.
    # shellcheck disable=SC2086
    POLL_PIDS=$(snni_start_poller "$RES" poll $PIDS)
    RC=0; wait "$APP_PID" || RC=$?
    # shellcheck disable=SC2086
    [ -n "$POLL_PIDS" ] && wait $POLL_PIDS 2>/dev/null
fi

rm -rf "$WORK"
snni_finish "$RES" "$T0" "$RC"
exit $RC
