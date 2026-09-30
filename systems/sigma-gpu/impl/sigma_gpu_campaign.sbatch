#!/bin/bash
#SBATCH --job-name=sigma
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=00:06:00
# Time limit 8x the measured worst case (31 to 46 s); longer requests delayed the s1 chain 13 h.
#SBATCH --output=/scratch/tmp/j_wred02/sigma_gpu/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/sigma_gpu/slurm-%x-%j.err
#
# SIGMA-GPU on the shared campaign harness. Full H200 (baseline's fixed 40 GiB VRAM pool OOMs a
# 2g.35gb MIG slice in 10 s); whole line on full hardware. Both parties on one card over loopback.
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
# SNNI_AR selects a recorder build other than the live one (opt-in, 2026-09-29): a line re-measured
# with the build it was first measured with, from $CAMP/recorder_builds/<build>/. RUN_META records
# it (snni_run_meta). Unset, the live attrib_resident as before.
AR="${SNNI_AR:-$CAMP/attrib_resident}"
BASE=/scratch/tmp/j_wred02/sigma_gpu

. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy

STEP="${STEP:?need STEP}"
IMG_STEP="${STEP%_pmrec}"; IMG_STEP="${IMG_STEP%_attrib}"
# p_seq<N>_* probe: runs an existing step's image at a different sequence length, no build. 197
# is ViT-Base/16's token count, but the workload here is only shapes (no model loaded), so this
# is a geometry diagnostic, never a waterfall row.
case "$IMG_STEP" in
    p_seq*_*) SNNI_SEQ="${IMG_STEP#p_seq}"; SNNI_SEQ="${SNNI_SEQ%%_*}"
              IMG_STEP="${IMG_STEP#p_seq${SNNI_SEQ}_}"
              echo "GEOMETRY_PROBE=1 n_seq=$SNNI_SEQ on image $IMG_STEP" ;;
esac


# One image serves the whole line. A step's lever is a rebuilt `sigma` binary bind-mounted over
# the one baked into the image, which is how the earlier campaign iterated without reshipping 8 GB
# per round. SIGMA_BIN names it; unset means the image's own binary, i.e. the stock program.
SIF="${SIF:-$BASE/sigma-gpu-s0_default.sif}"
# Fallback is for probes/refuted-attempts/baseline only (p_*, x_*, *_default); a waterfall step
# must run its own image or risk measuring the wrong binary (happened 2026-08-05, s2_threads_4).
if [ ! -r "$SIF" ]; then
    case "${STEP:-}" in
        p_*|x_*|*_default)
            SIF="$BASE/sigma-gpu-s0_default.sif" ;;
        *)
            echo "FAILED: step '${STEP:-}' has no image at $SIF, and a waterfall step may not fall" >&2
            echo "        back to the baseline: it would measure the baseline binary under this" >&2
            echo "        step's name. Build its image, or name one explicitly through SIF." >&2
            exit 2 ;;
    esac
fi
[ -r "$SIF" ] || { echo "no image at $SIF" >&2; exit 2; }
BIN="${SIGMA_BIN:-$BASE/bin/sigma-$IMG_STEP}"
# A diagnostic step falls back to the baseline binary, not the image's stock program (which has
# no pinned seed/gate). The fallback is loud, not silent: on this system the step lives in the
# binary (the image is only the CUDA runtime), so a missing binary must fail loudly, not silently
# measure a predecessor's program under this step's name.
if [ ! -r "$BIN" ]; then
    case "${STEP:-}" in
        p_*|x_*|*_default)
            BIN="$BASE/bin/sigma-s0_default" ;;
        *)
            echo "FAILED: step '${STEP:-}' has no binary at $BIN. On this system the image is only" >&2
            echo "        the CUDA runtime, so THE STEP IS THE BINARY and falling back would" >&2
            echo "        measure the baseline program under this step's name. Build it with" >&2
            echo "        build_step_binary.sbatch, or name one explicitly through SIGMA_BIN." >&2
            exit 2 ;;
    esac
fi
[ -r "$BIN" ] || {
    echo "FAILED: no measured binary at $BIN. The image's own binary is the STOCK program (no" >&2
    echo "        pinned seed, no gate observable, input zeroed), so running it would measure a" >&2
    echo "        different system under this step's name." >&2
    exit 2
}

RES="$BASE/results/$STEP${RUN:+/run$RUN}"
snni_prepare_results "$RES" || exit $?

export SIGMA_MODEL="${SIGMA_MODEL:-bert-base}"
export SIGMA_SEQ="${SIGMA_SEQ:-${SNNI_SEQ:-128}}"
# Pinned rather than inherited from the node, as everywhere in this campaign: thread count changes
# per-thread buffers and allocator arenas, which is to say it changes the peak itself.
export SIGMA_THREADS="${SIGMA_THREADS:-16}"

# Recorder depth is 2 (measured, p_hostdepth2b): depth 1 collapses the whole host decomposition
# into one allocator row (cpuMalloc, 97.2% of peak); depth 2 separates it into the program's own
# call sites. Set before snni_run_meta so RUN_META records pm_depth=2.
export SNNI_PM_DEPTH="${SNNI_PM_DEPTH:-2}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

snni_run_meta "$RES" \
    "image=$SIF" \
    "image_digest=$(sha256sum "$SIF" | awk '{print $1}')" \
    "sigma_binary=$([ -r "$BIN" ] && echo "$BIN" || echo "image default (STOCK)")" \
    "sigma_binary_md5=$([ -r "$BIN" ] && md5sum "$BIN" | awk '{print $1}' || echo na)" \
    "model=$SIGMA_MODEL n_seq=$SIGMA_SEQ n_layer=12 n_head=12 n_embd=768 bw=50 scale=12" \
    "threads=$SIGMA_THREADS" \
    "role=both online parties, sharing one device, loopback"

# Diagnostic switch to disable the freeze, which is itself a suspect here: runs died in recvBytes
# on a short read, the same LLAMA-online shape caused by the freeze poller interrupting a transfer.
# Unsets SNNI_PM_FREEZE (setting it to 0 doesn't work; poll_peak.sh tests presence).
if [ -n "${SNNI_DIAG_NO_FREEZE:-}" ]; then
    echo "DIAGNOSTIC: freeze disabled for this run (SNNI_DIAG_NO_FREEZE)"
    unset SNNI_PM_FREEZE
fi

# Device channel skips the freeze: cudarec's table comes from an atexit hook, unaffected by
# stopping the process, while host numbers all come from the clean channel. The freeze used to
# kill runs here (recvBytes assert, jobs 45848688/45853567) once s1_nowarmup's extra allocation
# latency exposed the same poller-interrupts-a-transfer bug LLAMA online had; now fixed
# (build_step_binary.sbatch's recvBytes patch), a combined single-run channel is possible.
#
# Opt-in only (SNNI_COMBINED=1), so published rows keep the arrangement they were measured with
# until a run confirms the combination works; avoids the split-run decomposition mismatch
# PROCEDURE warns about (a step's own evidence once showed a table sampled 3.5% below its peak).
if [ -n "${ATTRIB:-}" ] && [ -z "${SNNI_COMBINED:-}" ]; then
    echo "device channel: freeze disabled (the device table is recorded in the allocation hook and"
    echo "                written at exit; no host capture is published from this channel)"
    unset SNNI_PM_FREEZE
elif [ -n "${SNNI_COMBINED:-}" ]; then
    [ -n "${ATTRIB:-}" ] && [ -n "${PMREC:-}" ] || {
        echo "FAILED: SNNI_COMBINED=1 needs BOTH channels (ATTRIB=1 PMREC=1); with one of them" >&2
        echo "        it would be an ordinary single-channel run wearing the combined label." >&2
        exit 2; }
    echo "COMBINED CHANNEL: cudarec and pmrec in ONE run, freeze ON, declared"
    echo "combined_channel=1" >> "$RES/RUN_META"
fi

T0=$(date +%s)

# Recorder environment exported inside the container via an inner shell, not apptainer --env
# (which manages LD_PRELOAD itself in some configs and can silently drop it).
INNER="$(snni_pmrec_inner_env "$RES") export SIGMA_RESULTS=$RES"
# Device channel (cudarec) was previously never armed here since ATTRIB was unread (3 derives
# failed before noticed). SNNI_CUDAREC_OUT is left to the runner (per party, since two parties
# share one card). LD_PRELOAD is appended, not reassigned, so enabling both recorders doesn't
# silently drop one (an earlier version's second assignment replaced the first).
if [ -n "${ATTRIB:-}" ]; then
    if [ -n "${PMREC:-}" ]; then
        INNER="$INNER; export LD_PRELOAD=/ar/pmrec.so:$CAMP/attrib_device/cudarec.so SNNI_CUDAREC=1 SNNI_CUDAREC_TABLE=$RES/devtable_%d"
    else
        INNER="$INNER; export LD_PRELOAD=$CAMP/attrib_device/cudarec.so SNNI_CUDAREC=1 SNNI_CUDAREC_TABLE=$RES/devtable_%d"
    fi
fi
INNER="$INNER SIGMA_MODEL=$SIGMA_MODEL SIGMA_SEQ=$SIGMA_SEQ SIGMA_THREADS=$SIGMA_THREADS"
# keyBufSz is a per-model constant that doesn't scale with token count (16.83/20 GiB used at
# n_seq=128; overflowed at 197, killing p_seq197). Override is inert without this variable, so no
# published binary/row is affected.
[ -n "${SIGMA_KEYBUF_MB:-}" ] && INNER="$INNER SIGMA_KEYBUF_MB=$SIGMA_KEYBUF_MB"
INNER="$INNER; exec bash $BASE/run_sigma_measured.sh"

# Program writes reports relative to its read-only image directory; bound to real disk rather
# than --writable-tmpfs, since a memory-measurement job shouldn't let writes become shmem.
mkdir -p "$RES/container_output"

apptainer exec --nv \
    --bind "$RES":"$RES" \
    ${ATTRIB:+--bind "$CAMP/attrib_device":"$CAMP/attrib_device"} \
    --bind "$RES/container_output":/sigma/EzPC/GPU-MPC/experiments/sigma/output \
    --bind "$BASE/run_sigma_measured.sh":"$BASE/run_sigma_measured.sh" \
    --bind "$BIN":/sigma/EzPC/GPU-MPC/experiments/sigma/sigma \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

# BOTH PARTIES ARE THE SAME BINARY WITH THE SAME NAME, so no exe or cmdline match can tell them
# apart. The launcher writes its pids; we read them. That is the general rule this campaign
# arrived at after attaching a poller to a 398 MB launcher while the workers sat at 2.7 GB.
PIDS=""
for _ in $(seq 1 600); do
    PIDS=$(snni_worker_pids_from_files "$RES" 'party*.pid')
    [ "$(echo "$PIDS" | wc -w)" -ge 2 ] && break
    kill -0 "$APP_PID" 2>/dev/null || break
    sleep 0.1
done

if [ "$(echo "$PIDS" | wc -w)" -lt 2 ]; then
    echo "ERROR: fewer than two party pids appeared; this run has no usable peak" >&2
    RC=4
    wait "$APP_PID" 2>/dev/null
else
    # One leader freezes BOTH parties with a single kill. Stopping one peer of a two-party protocol
    # while the other runs is not a pause, it is a protocol fault.
    # shellcheck disable=SC2086
    POLL_PIDS=$(snni_start_poller "$RES" poll $PIDS)
    RC=0; wait "$APP_PID" || RC=$?
    # shellcheck disable=SC2086
    [ -n "$POLL_PIDS" ] && wait $POLL_PIDS 2>/dev/null
fi

# Gate observable is written by the launcher; comm volume and key size are checks beside it (not
# the gate itself), since the protocol is data-oblivious and neither should change with a lever.
{
    grep -ho 'Total Comm[^,]*' "$RES"/P0/models/*/evaluator.txt 2>/dev/null
    grep -ho 'Key size=[0-9]*' "$RES"/P0/models/*/dealer.txt 2>/dev/null
} > "$RES/invariants.txt" 2>/dev/null
cat "$RES/invariants.txt" 2>/dev/null

# Gate is structural (tier T1s): the value observable doesn't work here (output is 98,304 exact
# zeros, measured), but the transcript reproduces byte-exactly and moves only under structural
# changes (sequence length, not thread count) -- PROCEDURE's admission test.
#
# ONE TOKEN PER LINE, because `read_observable` takes the first token of each line and the raw
# lines begin with the word "Total".
{
    sed -n 's/^Total Comm=\([0-9]*\) B.*/COMM:\1/p' "$RES/invariants.txt"
    sed -n 's/^Key size=\([0-9]*\).*/KEYS:\1/p'     "$RES/invariants.txt"
} > "$RES/gate.txt" 2>/dev/null

# A STRUCTURAL OBSERVABLE CAN BE MISSING OR DEGENERATE TOO. Missing means the run never reported its
# transcript; degenerate here means a zero volume, which is what a run that exchanged nothing looks
# like rather than a run whose protocol is cheap.
if [ "$(wc -l < "$RES/gate.txt" 2>/dev/null || echo 0)" -lt 2 ]; then
    echo "FAILED: no transcript invariants, so this run has no gate observable" >&2
    [ "$RC" = 0 ] && RC=5
elif grep -qE '^(COMM|KEYS):0*$' "$RES/gate.txt" 2>/dev/null; then
    echo "FAILED: a transcript invariant is zero, i.e. this run exchanged nothing" >&2
    [ "$RC" = 0 ] && RC=6
fi

snni_finish "$RES" "$T0" "$RC"
exit $RC
