#!/bin/bash
#SBATCH --job-name=TODO_short_name
#SBATCH --partition=TODO_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=TODO_HH:MM:SS
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/TODO_system/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/TODO_system/slurm-%x-%j.err
#
# Template: copy to <workdir>/<system>.sbatch, fill every TODO. Shared parts belong in
# ../measure.sh or ../campaign.env. --exclusive and pinned threads are required.
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
AR=$CAMP/attrib_resident
BASE=$CAMP/TODO_system                          # TODO: this system's workdir

. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy

STEP="${STEP:?need STEP}"
# Channel suffixes are stripped to find the image: <step>_pmrec and <step>_attrib measure the same
# step through different instruments and must run the SAME binary.
IMG_STEP="${STEP%_pmrec}"; IMG_STEP="${IMG_STEP%_attrib}"

SIF="${SIF:-$BASE/sif/TODO_system-$IMG_STEP.sif}"   # TODO: image naming
[ -r "$SIF" ] || { echo "no image at $SIF" >&2; exit 2; }

RES="$BASE/results/$STEP${RUN:+/run$RUN}"
snni_prepare_results "$RES" || exit $?

# TODO: pin the thread count for this system, and say why this number.
export OMP_NUM_THREADS=TODO

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

snni_run_meta "$RES" \
    "image=$SIF" \
    "image_digest=$(sha256sum "$SIF" | awk '{print $1}')" \
    "omp_num_threads=$OMP_NUM_THREADS" \
    "model=TODO bert-base n_seq=128 ..." \
    "role=TODO which party, measured how"

# ---------------------------------------------------------------------------------------------
# TODO: launch the payload.
#
# Environment goes into an inner shell that execs the binary, not apptainer --env (which manages
# LD_PRELOAD itself in some configs and can silently drop it); exec keeps the pid stable for the poller.
T0=$(date +%s)
INNER="$(snni_pmrec_inner_env "$RES") export OMP_NUM_THREADS=$OMP_NUM_THREADS"
INNER="$INNER; exec TODO_binary TODO_args"

apptainer exec \
    --bind "$RES":"$RES" \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

# ---------------------------------------------------------------------------------------------
# TODO: identify the process(es) holding the memory. Use harness-written pid files if available;
# never pattern-match cmdline for a spawned child (it never carries the script name).
#
#   PIDS=$(snni_worker_pids_from_files "$RES" 'party*.pid')     multi-party, harness writes pids
#   PIDS=$(snni_find_payload "$APP_PID" TODO_binary_basename)   single process
PIDS=$(snni_find_payload "$APP_PID" TODO_binary_basename)

if [ -z "$PIDS" ]; then
    echo "ERROR: no payload process under $APP_PID; this run has no peak" >&2
    RC=4
else
    # shellcheck disable=SC2086
    POLL_PIDS=$(snni_start_poller "$RES" poll $PIDS)
    RC=0; wait "$APP_PID" || RC=$?
    # shellcheck disable=SC2086
    [ -n "$POLL_PIDS" ] && wait $POLL_PIDS 2>/dev/null
fi

# ---------------------------------------------------------------------------------------------
# TODO: the correctness gate. Write the observable to $RES/gate.txt, compared against the
# baseline's (tests whether this step changed anything, not whether the system is correct overall).
# Fail if missing; check size after hashing (network filesystem writeback can lag).
#
#   md5sum TODO_output_files > "$RES/gate.txt" || RC=5
#   [ -s "$RES/gate.txt" ] || { echo "FAILED: no gate observable" >&2; RC=5; }
TODO_gate_command

snni_finish "$RES" "$T0" "$RC"
exit $RC
