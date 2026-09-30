#!/bin/bash
#SBATCH --job-name=sgpu
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#
# Full H200 via --gres since 2026-08-01 (MIG slice: same lever +74.8% and -40.3% wall four hours
# apart, VRAM reproducible); not --exclusive (would block 8 GPUs). Whole line re-measured here.
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/shaft-gpu/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/shaft-gpu/slurm-%x-%j.err

# SHAFT-GPU unified harness: one image with every lever module; steps differ by environment (STEP,
# LEVER_* via --export). VRAM from in-container torch counters; host from the peak-triggered poller.
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
# SNNI_AR selects a recorder build other than the live one (opt-in, 2026-09-29): a line re-measured
# with the build it was first measured with, from $CAMP/recorder_builds/<build>/. RUN_META records
# it (snni_run_meta). Unset, the live attrib_resident as before.
AR="${SNNI_AR:-$CAMP/attrib_resident}"
BASE=$CAMP/shaft-gpu
STEP="${STEP:?need STEP}"

# Lever set declared in a file per step (previously a command-line argument to run_line.sh, so
# nothing on disk said what a step meant, and the shared run_step.sh once ran the baseline under
# three step names). A missing file is refused, not defaulted to "no levers" (same rule as SHAFT-CPU).
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

# Uses the shared harness, not a per-system copy of results layout/poll interval/recorder floor;
# what remains here is genuinely this system's: MIG allocation, device channel, pid discovery.
. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy
SIF="${SIF:-/scratch/tmp/j_wred02/shaft_gpu/shaft-gpu-estream.sif}"
RESULTS="$BASE/results/$STEP${RUN:+/run$RUN}"

# One directory per replicate: previously three chained replicates each displaced the last in
# logs/<step>, leaving one run's data under three successful exit codes.
snni_prepare_results "$RESULTS" || exit $?

# Baseline script is bound over the image's copy (carries the gate, identical across clean and
# attribution runs); its md5 is recorded since the image digest no longer fully determines the code.
BASELINE_SCRIPT="${BASELINE_SCRIPT:-$BASE/bert_instrumented_gpu.py}"

# A step may replace the driver script, not only the levers; the bind must use SHAFT_SCRIPT's
# name (previously fixed, so an override once bound correctly but ran python against a nonexistent
# file). Guarded so a mismatched name/script pair fails loudly rather than silently measuring the baseline.
SHAFT_SCRIPT="${SHAFT_SCRIPT:-bert_instrumented_gpu.py}"
if [ "$SHAFT_SCRIPT" != "bert_instrumented_gpu.py" ] && \
   [ "$(basename "$BASELINE_SCRIPT")" != "$SHAFT_SCRIPT" ]; then
    echo "ERROR: SHAFT_SCRIPT=$SHAFT_SCRIPT but BASELINE_SCRIPT=$BASELINE_SCRIPT." >&2
    echo "       The step names a driver it does not supply; the bind would deliver the baseline" >&2
    echo "       under that name and this run would measure the baseline. Set BASELINE_SCRIPT to" >&2
    echo "       the driver in the step declaration." >&2
    exit 2
fi
if [ ! -r "$BASELINE_SCRIPT" ]; then
    echo "ERROR: BASELINE_SCRIPT=$BASELINE_SCRIPT is not readable on the cluster." >&2
    echo "       Run tools/palma/deploy.sh for this system before submitting." >&2
    exit 2
fi
# Lever modules bound in from the campaign directory (same reason as the baseline script): what
# the image contains is not what this repository publishes; md5 recorded below.
LEVER_MODULES="matmul_limb_loop_gpu matmul_chunk_gpu embed_chunk_gpu graphop_markers_gpu plaintext_free_gpu"
# Generic-slot modules are bound too, or the script would import the image's own copy invisibly
# (RUN_META would then record the md5 of a file that never ran).
LEVER_MODULES="$LEVER_MODULES $(echo "${SHAFT_LEVERS:-}" | tr ',' ' ')"
LEVER_BINDS=()
for m in $LEVER_MODULES; do
    [ -f "$BASE/$m.py" ] && LEVER_BINDS+=(--bind "$BASE/$m.py:/root/shaft/$m.py")
done
for m in $(echo "${SHAFT_LEVERS:-}" | tr ',' ' '); do
    [ -f "$BASE/$m.py" ] || { echo "ERROR: SHAFT_LEVERS names $m but $BASE/$m.py does not exist" >&2; exit 2; }
done
mkdir -p "$RESULTS"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

{
  echo "STEP=$STEP"
  echo "DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "HOST=$(hostname)"
  echo "SIF=$SIF"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-na}"
  echo "GRES=${SLURM_JOB_GRES:-na}"
  echo "SHAFT_SCRIPT=${SHAFT_SCRIPT:-bert_instrumented_gpu.py}"
  echo "BASELINE_SCRIPT_MD5=$(md5sum "$BASELINE_SCRIPT" | awk '{print $1}')"
  for m in $LEVER_MODULES; do
      [ -f "$BASE/$m.py" ] && echo "MODULE_MD5_$m=$(md5sum "$BASE/$m.py" | awk '{print $1}')"
  done
  echo "SHAFT_LEVERS=${SHAFT_LEVERS:-<unset>}"
  echo "SHAFT_PARAM_STREAM=${SHAFT_PARAM_STREAM:-<unset>}"
  echo "SHAFT_EMBED_CHUNKS=${SHAFT_EMBED_CHUNKS:-<unset>}"
  echo "SHAFT_EMBED_STREAM=${SHAFT_EMBED_STREAM:-<unset>}"
  echo "SHAFT_LIMB_LOOP=${SHAFT_LIMB_LOOP:-<unset>}"
  echo "SHAFT_PLAINTEXT_FREE=${SHAFT_PLAINTEXT_FREE:-<unset>}"
  echo "SHAFT_MALLOC_TRIM=${SHAFT_MALLOC_TRIM:-<unset>}"
  echo "SHAFT_MATMUL_CHUNKS=${SHAFT_MATMUL_CHUNKS:-<unset>}"
  echo "SHAFT_DEV_ENCRYPT=${SHAFT_DEV_ENCRYPT:-<unset>}"
  echo "ATTRIBUTION=${ATTRIBUTION:-off}"
  # Exactly one of these may be set. A clean run has both off and is the only run whose peaks
  # and wall time may be published; an attribution run reports composition and nothing else.
  echo "ATTRIB_DEVICE=$([ -n "${ATTRIB:-}" ] && echo torch_state || echo off)"
  echo "ATTRIB_HOST=$([ -n "${MEMRAY:-}" ] && echo memray || { [ -n "${PMREC:-}" ] && echo pmrec_pagemap || echo off; })"
  snni_meta_optin
} > "$RESULTS/RUN_META"

# Two attribution channels together produced a deadlock costing four shim revisions, so the
# harness refuses the combination rather than relying on the caller.
N_ATTRIB=0
for v in "${ATTRIB:-}" "${MEMRAY:-}" "${PMREC:-}"; do [ -n "$v" ] && N_ATTRIB=$((N_ATTRIB + 1)); done
# SNNI_COMBINED_PROBE=1 (diagnostic) and SNNI_COMBINED=1 (steps, in use since 2026-08-01) both
# lift the pmrec+device refusal, which was inherited from a memray deadlock that pmrec.so's design
# (lock-free, no signal handler) likely avoids. Licensed by measurement: three probe runs (jobs
# 45560087/207/208) showed no deadlock, byte-identical gate and VRAM. Taken here rather than left
# at probe status since the whole line was already being re-measured for the machine move.
if [ "$N_ATTRIB" -gt 1 ]; then
    if [ -n "${PMREC:-}" ] && [ -n "${ATTRIB:-}" ] && [ -z "${MEMRAY:-}" ] \
       && { [ "${SNNI_COMBINED_PROBE:-}" = "1" ] || [ "${SNNI_COMBINED:-}" = "1" ]; }; then
        if [ "${SNNI_COMBINED:-}" = "1" ]; then
            echo "COMBINED CHANNEL: pmrec and the device snapshotter in ONE run, declared" \
                | tee -a "$RESULTS/RUN_META"
        else
            echo "COMBINED PROBE: pmrec and the device channel in ONE run; diagnostic, never a step" \
                | tee -a "$RESULTS/RUN_META"
        fi
    else
        echo "ERROR: more than one attribution channel set; they run separately" >&2
        exit 2
    fi
fi
cat "$RESULTS/RUN_META"
nvidia-smi -L 2>/dev/null || true

ENVARGS=()
for v in SHAFT_EMBED_CHUNKS SHAFT_EMBED_STREAM SHAFT_LIMB_LOOP SHAFT_MALLOC_TRIM SHAFT_MATMUL_CHUNKS SHAFT_DEV_ENCRYPT SHAFT_PLAINTEXT_FREE SHAFT_LEVERS SHAFT_PARAM_STREAM; do
    [ -n "${!v:-}" ] && ENVARGS+=(--env "$v=${!v}")
done
# SHAFT_INNER_ENV (borrowed from SHAFT-CPU) sets environment needed before the process starts,
# e.g. allocator tunables a Python module can't set late enough to matter; recorded in RUN_META.
for kv in ${SHAFT_INNER_ENV:-}; do ENVARGS+=(--env "$kv"); done
echo "SHAFT_INNER_ENV=${SHAFT_INNER_ENV:-<none>}" >> "$RESULTS/RUN_META"
echo "active levers: ${ENVARGS[*]:-<none, baseline>}"

# Python census (transplanted from shaft-cpu): host peak's top rows here are raw offsets in
# stripped binaries (28.9%/28.8%), unfixable by the C recorder since the owning object is Python's.
# Travels via the already-bound RESULTS directory; sitecustomize.py arms tracemalloc at interpreter start.
PYCENSUS_ON=""
if [ -n "${SNNI_PY_CENSUS:-}" ]; then
    _mod="$BASE/snni_pycensus.py"
    _arm="$BASE/snni_pycensus_sitecustomize.py"
    for _f in "$_mod" "$_arm"; do
        [ -r "$_f" ] || { echo "FAILED: SNNI_PY_CENSUS set but $_f is missing" >&2; exit 2; }
    done
    mkdir -p "$RESULTS/pycensus" || exit 2
    cp "$_mod" "$RESULTS/pycensus/snni_pycensus.py" || exit 2
    cp "$_arm" "$RESULTS/pycensus/sitecustomize.py" || exit 2
    PYCENSUS_ON=1
fi

# vit_* is a transfer check, not a waterfall row: binds a ViT checkpoint with everything else
# (levers, seeds, gate) unchanged. Only the two SHAFT systems load a real model; this line adds
# the device pool (19.232x VRAM reduction vs 1.365x host). See GEOMETRY-TRANSFER.md.
VIT_BIND=""
case "$STEP" in
    vit_*) VIT_DIR="${VIT_DIR:-/scratch/tmp/j_wred02/hf_vit/vit-base-patch16-224}"
           [ -r "$VIT_DIR/model.safetensors" ] || {
               echo "FAILED: no ViT checkpoint at $VIT_DIR" >&2; exit 2; }
           VIT_BIND=1
           echo "VIT_TRANSFER=1 checkpoint=$VIT_DIR"
           echo "VIT_TRANSFER=1 checkpoint=$VIT_DIR" >> "$RESULTS/RUN_META" ;;
esac

start_ts=$(date +%s)
apptainer exec --nv \
    --pwd /root/shaft \
    ${VIT_BIND:+--bind "$VIT_DIR":/vit} \
    --env HF_HOME=/root/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env SHAFT_SCRIPT="$SHAFT_SCRIPT" \
    --bind "$BASELINE_SCRIPT":"/root/shaft/$SHAFT_SCRIPT" \
    ${LEVER_BINDS[@]:+"${LEVER_BINDS[@]}"} \
    "${ENVARGS[@]}" \
    ${ATTRIB:+--bind /scratch/tmp/j_wred02/snni_campaign/attrib_gpu:/attrib} \
    ${ATTRIB:+--env PYTHONPATH=/attrib} \
    ${ATTRIB:+--env SNNI_ATTRIB=1} \
    ${MEMRAY:+--bind /scratch/tmp/j_wred02/snni_campaign/attrib_memray:/attrib_memray} \
    ${MEMRAY:+--bind /scratch/tmp/j_wred02/snni_campaign/memray:/memray} \
    ${MEMRAY:+--env PYTHONPATH=/attrib_memray:/memray} \
    ${MEMRAY:+--env SNNI_MEMRAY=1} \
    ${MEMRAY:+--env SNNI_MEMRAY_DIR=/results} \
    ${PMREC:+--bind "$AR":/ar} \
    ${PMREC:+--env LD_PRELOAD=/ar/pmrec.so} \
    ${PMREC:+--env SNNI_PM_TABLE=/results/pmtable_%d} \
    ${PMREC:+--env SNNI_PM_MIN=${SNNI_PM_MIN:?campaign.env must supply the recorder floor}} \
    ${PMREC:+--env SNNI_PM_FULLSCAN=${SNNI_PM_FULLSCAN:?campaign.env must supply the scan mode}} \
    ${PYCENSUS_ON:+--env PYTHONPATH=/results/pycensus} \
    ${PYCENSUS_ON:+--env SNNI_PY_CENSUS=1} \
    ${PYCENSUS_ON:+--env SNNI_PY_CENSUS_DIR=/results} \
    ${SNNI_ATTRIB_MODE:+--env SNNI_ATTRIB_MODE=${SNNI_ATTRIB_MODE}} \
    --bind "$RESULTS:/results" \
    "$SIF" bash /root/shaft/run_gpu.sh /results > "$RESULTS/container_output.log" 2>&1 &
APP_PID=$!

# Poller runs outside the container (apptainer shares the PID namespace), adding a composition
# snapshot at each new maximum. Whole party set goes into one snni_start_poller call, not one per
# pid: calling it per pid once made both parties leader=1, doubling replicates in one directory
# (46.6% spread). Pids come from run_gpu.sh's party<N>.pid, not cmdline matching (SPAWN children
# never show the script name and would attach to the 398 MB launcher).
# Result subdirectory is discovered by glob (run_gpu.sh names it n<layers>_seq<len>), not
# hardcoded: this file once had n12_seq128 hardcoded in three places, so a ViT run (seq 197)
# succeeded but reported no party process. Discovered rather than predicted from SHAFT_MAX_LENGTH,
# which isn't a shell variable here and would have re-defaulted to 128.
RESDIR=""
NPARTIES=2
PIDS=""
for _ in $(seq 1 1200); do
    n=$(ls "$RESULTS"/n*_seq*/party*.pid 2>/dev/null | wc -l)
    [ "$n" -ge "$NPARTIES" ] && break
    kill -0 "$APP_PID" 2>/dev/null || break
    sleep 0.5
done
# Ordered by party number, so party 0 leads. Only the leader's capture is atomic, so which party
# leads is fixed rather than incidental: the published decomposition is always party 0's.
# One directory or the run is not what it says it is: `run_gpu.sh` makes exactly one per run.
_rd=$(ls -d "$RESULTS"/n*_seq*/ 2>/dev/null | head -1)
if [ -n "$_rd" ]; then RESDIR=$(basename "$_rd"); fi
if [ "$(ls -d "$RESULTS"/n*_seq*/ 2>/dev/null | wc -l)" -gt 1 ]; then
    echo "ERROR: more than one n*_seq* directory under $RESULTS; one run makes one." >&2
    exit 2
fi
echo "RESDIR=${RESDIR:-<none>}" >> "$RESULTS/RUN_META"
for pf in "$RESULTS/$RESDIR/party0.pid" "$RESULTS/$RESDIR/party1.pid"; do
    [ -r "$pf" ] || continue
    p=$(tr -dc '0-9' < "$pf")
    [ -n "$p" ] && [ -d "/proc/$p" ] && PIDS="$PIDS $p"
done
PIDS="${PIDS# }"
POLL_PIDS=""
if [ -z "$PIDS" ]; then
    echo "ERROR: no party process under $APP_PID; this run has no peak" >&2
else
    # Census triggered by the poller's high-water detection (15ms samples), not a clock: on
    # shaft-cpu a time-driven census sat at 79% of the polled peak.
    export SNNI_CENSUS_SIGNAL="${PYCENSUS_ON:+USR1}"

    echo "party set: [$PIDS]" >> "$RESULTS/poller_attach.log"
    # shellcheck disable=SC2086
    POLL_PIDS=$(snni_start_poller "$RESULTS" poll $PIDS)
fi


wait $APP_PID
EC=$?
# shellcheck disable=SC2086
[ -n "$POLL_PIDS" ] && wait $POLL_PIDS 2>/dev/null
end_ts=$(date +%s)

echo $(( end_ts - start_ts )) > "$RESULTS/wall_seconds.txt"
D="$RESULTS/$RESDIR"
# The GPUPEAK marker is `GPUPEAK|party=N|peak_alloc_kb=A|peak_reserved_kb=R`, so its fields carry
# names, not bare numbers. Match the named field; both parties are reported and the binding value
# is the max, because both must fit.
VR=$(grep -ohE 'peak_reserved_kb=[0-9]+' "$D/all_markers.txt" 2>/dev/null \
     | cut -d= -f2 | sort -n | tail -1)
HR=$(awk -F'|' '/^MEM\|/{if($6+0>m)m=$6+0} END{print m+0}' "$D/all_markers.txt" 2>/dev/null)
{
  echo "step=$STEP  exit=$EC  wall=$(( end_ts - start_ts ))s"
  echo "vram_reserved_peak_kb=${VR:-NA}"
  echo "host_marker_peak_kb=${HR:-NA}"
} > "$RESULTS/PEAK.txt"

# The gate observable, split one value per line so a diff points at which value moved rather than
# at a single very long line.
grep -h '^GATE|logits|' "$RESULTS/container_output.log" "$D/stdout.log" 2>/dev/null \
    | head -1 | cut -d'|' -f3- | tr ' ' '\n' | grep -v '^$' > "$RESULTS/gate.txt"
# A lever that never attached publishes the baseline under its name unless checked, via each
# module's stderr announcement. Only markers that appear exactly when active are checked here
# (embed_chunk's marker also fires at chunks=1, a no-op, so it's excluded).
check_lever() {  # <env var> <active test> <marker>
    local v="$1" want="$2" mark="$3" val="${!1:-}"
    [ "$val" = "$want" ] || return 0
    # Both files checked: the script's stderr goes to all_markers.txt, reaching
    # container_output.log only if run_gpu.sh's grep happens to match (these two pass by luck, via CUDA).
    if ! grep -qs "$mark" "$RESULTS/container_output.log" "$D/all_markers.txt"; then
        echo "ERROR: $v=$val was requested but '$mark' never appeared in the run output." >&2
        echo "       The lever did not load, so this run measures the BASELINE under its name." >&2
        return 1
    fi
    echo "lever $v=$val confirmed active" | tee -a "$RESULTS/RUN_META"
}
check_lever SHAFT_LIMB_LOOP 1 "matmul_limb_loop_gpu: patched" || exit 5
check_lever SHAFT_PLAINTEXT_FREE 1 "plaintext_free_gpu: patched" || exit 5
# Generic-slot modules must announce "<module>: patched" on stderr, checked automatically so a
# new lever needs no added line. Both files are searched, since container_output.log only mirrors
# all_markers.txt lines matching run_gpu.sh's own error grep (a real gap that once refused a valid lever).
for m in $(echo "${SHAFT_LEVERS:-}" | tr ',' ' '); do
    if ! grep -qs "$m: patched" "$RESULTS/container_output.log" "$D/all_markers.txt"; then
        echo "ERROR: SHAFT_LEVERS requested $m but '$m: patched' never appeared in the output." >&2
        echo "       The lever did not patch, so this run measures the BASELINE under its name." >&2
        exit 5
    fi
    echo "lever $m confirmed active" | tee -a "$RESULTS/RUN_META"
done

# Recorder attachment is verified by its effect (a table per process), not by having been
# requested, since apptainer manages LD_PRELOAD itself at some sites and can silently drop it.
if [ -n "${PMREC:-}" ]; then
    NT=$(ls "$RESULTS"/pmtable_* 2>/dev/null | wc -l)
    echo "pmrec tables written: $NT" | tee -a "$RESULTS/RUN_META"
    if [ "$NT" -eq 0 ]; then
        echo "ERROR: PMREC=1 but no pmtable_<pid> exists; the recorder never attached, so this" >&2
        echo "       run has NO host decomposition. Not a slow run: an empty one." >&2
        exit 4
    fi
fi

if [ ! -s "$RESULTS/gate.txt" ]; then
    rm -f "$RESULTS/gate.txt"
    echo "WARNING: no GATE|logits line; this step has no correctness gate" >&2
fi
cat "$RESULTS/PEAK.txt"
exit $EC
