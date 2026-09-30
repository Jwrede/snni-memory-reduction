#!/bin/bash
#SBATCH --job-name=scpu
#SBATCH --partition=express
#SBATCH --exclude=r07n04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/shaft-cpu/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/shaft-cpu/slurm-%x-%j.err
#
# SHAFT-CPU on the shared campaign harness (from tools/palma/templates/system.sbatch).
# --exclusive and pinned threads required.
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
# SNNI_AR selects a recorder build other than the live one (opt-in, 2026-09-29): a line re-measured
# with the build it was first measured with, from $CAMP/recorder_builds/<build>/. RUN_META records
# it (snni_run_meta). Unset, the live attrib_resident as before.
AR="${SNNI_AR:-$CAMP/attrib_resident}"
BASE=$CAMP/shaft-cpu

. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy

# Recorder depth is 1, a measured decision: deeper frames stay inside the same third-party
# library (Python's native stack never passes through measured code) and cost +16% wall time for
# nothing. Objects are named instead by the Python census (+12% cost).
export SNNI_PM_DEPTH="${SNNI_PM_DEPTH:-1}"

STEP="${STEP:?need STEP}"

# One image serves every step (env-gated: a lever is a bind-mounted Python module, imported only
# when SHAFT_LEVERS names it). A step's code is identified by lever set plus module md5s in
# RUN_META, not by a per-step image name.
SIF="${SIF:-$BASE/sif/shaft-cpu-s0_default.sif}"
[ -r "$SIF" ] || { echo "no image at $SIF" >&2; exit 2; }

# ---------------------------------------------------------------------------------------------
# Lever set is declared in a file per step (the shared harness has no lever plumbing since the
# three lever mechanisms across systems aren't unifiable). A missing file is refused, not
# defaulted to "no levers": that default would silently measure the baseline under a lever's name.
STEP_BASE="${STEP%_pmrec}"; STEP_BASE="${STEP_BASE%_attrib}"
STEP_ENV="$BASE/stepenv/$STEP_BASE.env"
if [ -r "$STEP_ENV" ]; then
    # Parsed before sourced: a syntax error used to be sourced anyway, leaving every SHAFT_*
    # variable unset and measuring the baseline under the step's name (cost a run on
    # x3_low_cpu_mem_load, an unbalanced backtick in a comment).
    if ! bash -n "$STEP_ENV" 2>/tmp/stepenv_syntax.$$; then
        echo "ERROR: $STEP_ENV is not valid shell:" >&2
        cat /tmp/stepenv_syntax.$$ >&2; rm -f /tmp/stepenv_syntax.$$
        echo "       Sourcing it would leave every lever unset and measure the baseline under" >&2
        echo "       this step's name." >&2
        exit 2
    fi
    rm -f /tmp/stepenv_syntax.$$
    # shellcheck disable=SC1090
    if ! . "$STEP_ENV"; then
        echo "ERROR: sourcing $STEP_ENV failed; refusing to run with a half-applied declaration" >&2
        exit 2
    fi
else
    echo "ERROR: no lever declaration at $STEP_ENV." >&2
    echo "       A step whose lever set is not declared would run as the baseline and publish" >&2
    echo "       that number under this step's name. Write the file, even if it is empty." >&2
    exit 2
fi
export SHAFT_LEVERS="${SHAFT_LEVERS:-}"
# SHAFT_INNER_ENV declares environment needed before the process starts (a Python lever can't
# set allocator tunables late enough to matter); recorded in RUN_META as part of the published lever.
export SHAFT_INNER_ENV="${SHAFT_INNER_ENV:-}"

RES="$BASE/results/$STEP${RUN:+/run$RUN}"
snni_prepare_results "$RES" || exit $?

# Thread count pinned to 8 (this system's own run.sh default); measured to move wall time only
# slightly (6.2s in 224.5s) and the peak not at all versus 48 threads.
export OMP_NUM_THREADS=8

# The measured script and the lever modules are BOUND OVER the image's copies, so the published
# file is the one that ran. Their md5s go into RUN_META because the image digest no longer fully
# determines the executed code.
SCRIPT="${SHAFT_SCRIPT:-$BASE/bert_instrumented_cpu.py}"
RUNNER="${SHAFT_RUNNER:-$BASE/run_cpu.sh}"
[ -r "$SCRIPT" ] || { echo "no measured script at $SCRIPT" >&2; exit 2; }
[ -r "$RUNNER" ] || { echo "no runner at $RUNNER" >&2; exit 2; }

# Lever modules: every *_cpu.py in the lever directory is bound in, and SHAFT_LEVERS names which
# of them the run imports. Binding one that is not named changes nothing -- an unimported module
# is a file on a filesystem -- while naming one that is not bound would fail loudly at import.
LEVER_BINDS=()
LEVER_MD5=()
for m in "$BASE"/levers/*.py; do
    [ -f "$m" ] || continue
    b=$(basename "$m")
    LEVER_BINDS+=(--bind "$m:/root/shaft/$b")
    LEVER_MD5+=("lever_md5_${b%.py}=$(md5sum "$m" | awk '{print $1}')")
done

# Python census (diagnostic, SNNI_PY_CENSUS=1) names objects the C recorder can only address
# (measured that Python's native stack never passes through measured code). Travels via $RES,
# already bound, rather than a new bind point in an image with no writable overlay.
PYCENSUS_PATH=""
if [ -n "${SNNI_PY_CENSUS:-}" ]; then
    _mod="$BASE/snni_pycensus.py"
    _arm="$BASE/snni_pycensus_sitecustomize.py"
    for _f in "$_mod" "$_arm"; do
        [ -r "$_f" ] || { echo "FAILED: SNNI_PY_CENSUS set but $_f is missing" >&2; exit 2; }
    done
    mkdir -p "$RES/pycensus" || exit 2
    # The module itself, plus a `sitecustomize.py` whose only job is to import it at interpreter
    # start. tracemalloc has to be armed BEFORE the program allocates, and a later import would
    # see none of what it is meant to measure.
    cp "$_mod" "$RES/pycensus/snni_pycensus.py" || exit 2
    cp "$_arm" "$RES/pycensus/sitecustomize.py" || exit 2
    PYCENSUS_PATH="$RES/pycensus"
fi

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

snni_run_meta "$RES" \
    "image=$SIF" \
    "image_digest=$(sha256sum "$SIF" | awk '{print $1}')" \
    "omp_num_threads=$OMP_NUM_THREADS" \
    "model=andeskyl/bert-base-cased-sst2 n_layers=12 seq_len=128 batch=1" \
    "role=2-party CrypTen MPC, both parties on this node's CPU, published per party" \
    "script_md5=$(md5sum "$SCRIPT" | awk '{print $1}')" \
    "runner_md5=$(md5sum "$RUNNER" | awk '{print $1}')" \
    "levers=${SHAFT_LEVERS:-<none, baseline>}" \
    "py_census=${SNNI_PY_CENSUS:-off} py_census_trace=${SNNI_PY_CENSUS_TRACE:-off}" \
    "inner_env=${SHAFT_INNER_ENV:-<none>}" \
    ${LEVER_MD5[@]:+"${LEVER_MD5[@]}"}

# ---------------------------------------------------------------------------------------------
# Launch.
#
# The environment goes into an INNER shell that then runs the payload, rather than through
# `apptainer --env`: apptainer manages LD_PRELOAD itself in some site configurations, and a
# preload silently dropped gives an attribution run with no recorder and no error.
T0=$(date +%s)
INNER="$(snni_pmrec_inner_env "$RES") export OMP_NUM_THREADS=$OMP_NUM_THREADS"
INNER="$INNER MKL_NUM_THREADS=$OMP_NUM_THREADS"
# vit_* is a transfer check, not a waterfall row: binds a ViT checkpoint (compute nodes have no
# network) with everything else unchanged, so the numbers transfer against an unchanged lever set.
# Only this system loads a real model; see GEOMETRY-TRANSFER.md.
case "$STEP" in
    vit_*) VIT_DIR="${VIT_DIR:-/scratch/tmp/j_wred02/hf_vit/vit-base-patch16-224}"
           [ -r "$VIT_DIR/model.safetensors" ] || {
               echo "FAILED: no ViT checkpoint at $VIT_DIR" >&2; exit 2; }
           VIT_BIND=1
           echo "VIT_TRANSFER=1 checkpoint=$VIT_DIR" ;;
esac
INNER="$INNER HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
INNER="$INNER SHAFT_N_LAYERS=12 SHAFT_MAX_LENGTH=128"
INNER="$INNER SHAFT_LEVERS='${SHAFT_LEVERS:-}'"
if [ -n "$PYCENSUS_PATH" ]; then
    # Prepended, never replacing: the image sets its own PYTHONPATH and dropping it would change
    # what the measured program imports, which is the one thing a diagnostic may not do.
    INNER="$INNER PYTHONPATH=$PYCENSUS_PATH:\${PYTHONPATH:-} SNNI_PY_CENSUS=1 SNNI_PY_CENSUS_DIR=$RES"
    INNER="$INNER${SNNI_PY_CENSUS_TRACE:+ SNNI_PY_CENSUS_TRACE=$SNNI_PY_CENSUS_TRACE}"
fi
INNER="$INNER${SHAFT_INNER_ENV:+ $SHAFT_INNER_ENV}"
INNER="$INNER; exec bash /root/shaft/run_cpu.sh $RES"

apptainer exec \
    --pwd /root/shaft \
    --bind "$RES":"$RES" \
    ${VIT_BIND:+--bind "$VIT_DIR":/vit} \
    --bind "$SCRIPT":/root/shaft/bert_instrumented_cpu.py \
    --bind "$RUNNER":/root/shaft/run_cpu.sh \
    ${LEVER_BINDS[@]:+"${LEVER_BINDS[@]}"} \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/container_output.log" 2>&1 &
APP_PID=$!

# ---------------------------------------------------------------------------------------------
# Identify the processes that HOLD THE MEMORY.
#
# Asks the harness for pids rather than pattern-matching cmdline: CrypTen's spawned children
# never show the script name, so a cmdline filter would attach to the 398 MB launcher instead of
# the 2.7 GB parties (measured on the GPU sibling).
#
# Both parties must publish before any poller starts: the freeze leader stops the whole set with
# one kill, so starting early would leave an unpublished peer running while its partner stops.
NPARTIES=2
PIDS=""
for _ in $(seq 1 1200); do
    n=$(ls "$RES"/party*.pid 2>/dev/null | wc -l)
    [ "$n" -ge "$NPARTIES" ] && break
    kill -0 "$APP_PID" 2>/dev/null || break
    sleep 0.5
done
# Ordered by party number, so party 0 is first and therefore the freeze LEADER. Only the leader's
# capture is atomic (see MANIFEST), so which party leads must be fixed rather than incidental:
# the published decomposition is always party 0's.
for pf in "$RES"/party0.pid "$RES"/party1.pid; do
    [ -r "$pf" ] || continue
    p=$(tr -dc '0-9' < "$pf")
    [ -n "$p" ] && [ -d "/proc/$p" ] && PIDS="$PIDS $p"
done
PIDS="${PIDS# }"

if [ -z "$PIDS" ]; then
    echo "ERROR: no party process under $APP_PID; this run has no peak" >&2
    RC=4
    wait "$APP_PID" 2>/dev/null
else
    # Census trigger is the poller's own high-water detection, not a clock: a prior 20s-cadence
    # shell watcher missed peaks (last census sat at 79% of the polled peak).
    export SNNI_CENSUS_SIGNAL="${SNNI_PY_CENSUS:+USR1}"

    echo "party set: [$PIDS]" >> "$RES/poller_attach.log"
    # shellcheck disable=SC2086
    POLL_PIDS=$(snni_start_poller "$RES" poll $PIDS)


    RC=0; wait "$APP_PID" || RC=$?
    # shellcheck disable=SC2086
    [ -n "$POLL_PIDS" ] && wait $POLL_PIDS 2>/dev/null
fi

# ---------------------------------------------------------------------------------------------
# The correctness gate, and the checks that say an instrument or a lever actually attached.
#
# Each of these turns a silent failure into a failed run. Every one of them exists because the
# failure it catches produces a plausible number rather than an error.
MARK="$RES/all_markers.txt"

# 1. THE SEED TOOK. A seed that silently failed to take looks exactly like one that worked, and
#    this system had no gate at all until it was pinned. Both parties must announce.
NSEED=$(grep -c '^SEEDED|party=' "$MARK" 2>/dev/null)
NSEED=${NSEED:-0}
if [ "$NSEED" -lt "$NPARTIES" ]; then
    echo "ERROR: $NSEED SEEDED| markers for $NPARTIES parties. The seeds did not take, so the" >&2
    echo "       shares differ per run and this run cannot be compared against anything." >&2
    RC=6
fi

# 2. Every requested lever loaded: an unimported module would be a successful zero-delta run,
#    publishing the baseline under the lever's name.
_LV="${SHAFT_LEVERS:-}"
for lv in ${_LV//,/ }; do
    if ! grep -q "^LEVER|$lv|patched" "$MARK" 2>/dev/null; then
        echo "ERROR: lever '$lv' was requested but never announced itself." >&2
        echo "       This run measures the BASELINE under a lever's name." >&2
        RC=5
    else
        echo "lever $lv confirmed active" | tee -a "$RES/RUN_META"
    fi
done

# 3. THE RECORDER ATTACHED. apptainer manages LD_PRELOAD itself at some sites, so the preload is
#    verified by its EFFECT -- a table per measured process -- rather than by having been asked
#    for. A run with no table has no decomposition; not a slow run, an empty one.
if [ -n "${PMREC:-}" ]; then
    NT=$(ls "$RES"/pmtable_* 2>/dev/null | wc -l)
    echo "pmrec tables written: $NT" | tee -a "$RES/RUN_META"
    if [ "$NT" -lt "$NPARTIES" ]; then
        echo "ERROR: PMREC=1 but only $NT pmtable_<pid> for $NPARTIES parties." >&2
        RC=4
    fi
fi

# 4. THE GATE OBSERVABLE EXISTS. A run that produced no gate output is not a run with a passing
#    gate. Split one value per line so a diff points at which value moved rather than at a single
#    very long line.
grep -h '^GATE|logits|' "$RES/stdout.log" "$RES/container_output.log" 2>/dev/null \
    | head -1 | cut -d'|' -f3- | tr ' ' '\n' | grep -v '^$' > "$RES/gate.txt"
if [ ! -s "$RES/gate.txt" ]; then
    rm -f "$RES/gate.txt"
    echo "ERROR: no GATE|logits line; this run has no correctness gate" >&2
    RC=7
fi

snni_finish "$RES" "$T0" "$RC"
exit $RC
