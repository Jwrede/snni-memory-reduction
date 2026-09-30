#!/bin/bash
#SBATCH --job-name=moaigpu
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/slurm-%x-%j.err
#
# MOAI-GPU under the shared harness (../harness/measure.sh). Not --exclusive: --gres isolates the
# H200; 110 GB memory request for 3.2 GiB host staging.
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
AR=$CAMP/attrib_resident
AD=$CAMP/attrib_device
BASE=$CAMP/moai-gpu

. "$CAMP/harness/measure.sh" || exit 2
snni_require_policy

STEP="${STEP:?need STEP}"
IMG_STEP="${STEP%_pmrec}"; IMG_STEP="${IMG_STEP%_attrib}"

SIF="${SIF:-$BASE/sif/moai-gpu-$IMG_STEP.sif}"
[ -r "$SIF" ] || { echo "no image at $SIF" >&2; exit 2; }

RES="$BASE/results/$STEP${RUN:+/run$RUN}"
snni_prepare_results "$RES" || exit $?

# Pinned at 8 (matches the retired campaign, for cross-check comparability); thread count changes
# per-thread buffers/arenas and thus the peak, so it is pinned rather than node-inherited.
export OMP_NUM_THREADS=8

# Host recorder depth 2 (distinct from device SNNI_CUDAREC_DEPTH below): at depth 1 the largest
# host row is a generic vector-growth function (65.3% of peak, no owner); depth 2 separates it into
# four `phantom::arith::NTT::NTT` sites (p_hostdepth2, 2026-08-10). Set before snni_run_meta.
export SNNI_PM_DEPTH="${SNNI_PM_DEPTH:-2}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

# Device channel is cudarec, not torch: measure.sh's `attrib_device=torch_state` is wrong here but
# is shared code, so the correct value is recorded alongside instead of editing it for one system.
snni_run_meta "$RES" \
    "image=$SIF" \
    "image_digest=$(sha256sum "$SIF" | awk '{print $1}')" \
    "omp_num_threads=$OMP_NUM_THREADS" \
    "model=BERT-base ONE encoder layer n_seq=128 batch=256 logN=16" \
    "role=single server process, non-interactive FHE" \
    "attrib_device_actual=$([ -n "${ATTRIB:-}" ] && echo cuda_alloc || echo off)" \
    "gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | head -1)"

# ---------------------------------------------------------------------------------------------
# Launch. The environment goes into an INNER shell that then execs the binary rather than through
# `apptainer --env`, because apptainer manages LD_PRELOAD itself in some site configurations and a
# silently dropped preload gives an attribution run with no recorder and no error.
INNER="$(snni_pmrec_inner_env "$RES")"
if [ -n "${ATTRIB:-}" ]; then
    # Device channel (cudarec.so). SNNI_CUDAREC_DEPTH defaults to 2: depth 1 keys on an allocation
    # helper that owns nothing (49.3% of peak, unattributable); depth 2 splits it into real owners,
    # measured to not perturb the peak (+0.15% wall, byte-identical peak_device_bytes).
    #
    # LD_PRELOAD is APPENDED when both channels are on, not assigned (2026-08-21 fix): the old code
    # let cudarec's assignment silently replace pmrec's, producing a run that looked combined and
    # wasn't -- the real (and only) reason this system used to run two runs per step. A later false
    # alarm (job 46345811, host peak 22x published) traced to an unrelated cause: a diagnostic step
    # id that didn't match the SPILL_DEFAULT case pattern below, so its ciphertext bank silently
    # parked in host RAM instead of on disk. Opt-in via SNNI_COMBINED so no published row changes
    # until a run confirms the combination.
    if [ -n "${PMREC:-}" ]; then
        [ -n "${SNNI_COMBINED:-}" ] || {
            echo "FAILED: ATTRIB and PMREC are both set but SNNI_COMBINED is not. Until 2026-08-21" >&2
            echo "        this combination silently ran ONE recorder. Set SNNI_COMBINED=1 to ask" >&2
            echo "        for the combined channel deliberately." >&2
            exit 2; }
        echo "COMBINED CHANNEL: cudarec and pmrec in ONE run, freeze ON, declared"
        echo "combined_channel=1" >> "$RES/RUN_META"
        INNER="$INNER export LD_PRELOAD=/ar/pmrec.so:$AD/cudarec.so SNNI_CUDAREC=1 SNNI_CUDAREC_OUT=$RES SNNI_CUDAREC_TABLE=$RES/devtable_%d SNNI_CUDAREC_DEPTH=${SNNI_CUDAREC_DEPTH:-2};"
    else
        INNER="$INNER export LD_PRELOAD=$AD/cudarec.so SNNI_CUDAREC=1 SNNI_CUDAREC_OUT=$RES SNNI_CUDAREC_TABLE=$RES/devtable_%d SNNI_CUDAREC_DEPTH=${SNNI_CUDAREC_DEPTH:-2};"
    fi
fi
# SPILL=1 enables the write-through disk bank (bound path must be inside $RES; elsewhere on
# /scratch is invisible in the container and silently falls back to host RAM). Derived from the
# step id, not hand-set, after a missing SPILL=1 cost a 66-minute run with misleading peaks
# (s12_ffn_threads_late, 2026-08-19). SPILL may still be set explicitly to override.
case "$IMG_STEP" in
    s5_*|s6_*|s7_*|s8_*|s9_*|s1[0-9]_*|s2[0-9]_*) SPILL_DEFAULT=1 ;;
    *)                                            SPILL_DEFAULT=0 ;;
esac
if [ "${SPILL:-$SPILL_DEFAULT}" = "1" ]; then
    mkdir -p "$RES/spill" || { echo "FAILED: cannot create the spill directory $RES/spill" >&2; exit 4; }
    INNER="$INNER export MOAI_SPILL_DIR=$RES/spill;"
    echo "SPILL_DIR=$RES/spill (SPILL=${SPILL:-$SPILL_DEFAULT}, derived from step id)"
fi
INNER="$INNER export OMP_NUM_THREADS=$OMP_NUM_THREADS"
INNER="$INNER; export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\${LD_LIBRARY_PATH:-}"
INNER="$INNER; cd /moai/build && exec ./test"

T0=$(date +%s)
# --writable-tmpfs: the program writes its timing CSV next to the (read-only) build directory.
# Per-run and discarded, so it changes nothing the measurement reads.
apptainer exec --nv --writable-tmpfs \
    --bind "$RES":"$RES" \
    --bind "$AD":"$AD" \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

# Matched on /proc/<pid>/exe, not cmdline: the wrapper and inner shell both carry `test` there too.
PIDS=$(snni_find_payload "$APP_PID" test)

if [ -z "$PIDS" ]; then
    echo "ERROR: no payload process under $APP_PID; this run has no peak" >&2
    RC=4
    wait "$APP_PID" 2>/dev/null
else
    POLL_PIDS=$(snni_start_poller "$RES" poll $PIDS)
    RC=0; wait "$APP_PID" || RC=$?
    # shellcheck disable=SC2086
    [ -n "$POLL_PIDS" ] && wait $POLL_PIDS 2>/dev/null
fi

# The program's own exit code, recorded before the guards below can overwrite RC with their own.
PROG_RC=$RC
echo "$PROG_RC" > "$RES/program_exit_code.txt"

# Disk bank is run-scratch (tens of GB, would otherwise be tarred); its size is the one number worth
# publishing before deletion.
if [ -d "$RES/spill" ]; then
    du -sb "$RES/spill" 2>/dev/null | awk '{print "SPILL_BYTES=" $1}' | tee "$RES/spill_bytes.txt"
    rm -rf "$RES/spill"
fi
if [ "$PROG_RC" -gt 128 ]; then
    echo "program died on signal $((PROG_RC - 128))" | tee -a "$RES/program_exit_code.txt" >&2
elif [ "$PROG_RC" -ne 0 ]; then
    echo "program exited $PROG_RC" >&2
fi

# Device peak and phase markers, emitted by the binary itself. GPUPEAK is the driver-maintained
# high-water mark (make_steps.py reads it), exact and unmissable, unlike a sampler.
mkdir -p "$RES/logs"
grep -E "^(GPUPEAK|HOSTPEAK)\|" "$RES/stdout.log" > "$RES/logs/all_markers.txt" 2>/dev/null
grep -E "^MEM\|" "$RES/stdout.log" > "$RES/markers.log" 2>/dev/null
grep -E "Modulus chain index" "$RES/stdout.log" > "$RES/levels.txt" 2>/dev/null

# Correctness gate is structural here (tier T1s, MANIFEST.md): no VALUE observable is stable across
# seeds, but the modulus-chain trace is byte-identical and was shown to discriminate
# (off-path-runs/p_level_discriminate*). Position-tagged so `read_observable`'s first-token read
# stays order-sensitive.
awk '{ n++; printf "L%02d:%s\n", n, $NF }' "$RES/levels.txt" > "$RES/gate.txt" 2>/dev/null
NLEV=$(wc -l < "$RES/gate.txt" 2>/dev/null || echo 0)
if [ "$NLEV" -lt 12 ]; then
    echo "FAILED: level trace has $NLEV of the 12 transitions a complete run emits, so this run" >&2
    echo "        did not finish the layer and its trace is a prefix rather than an observable" >&2
    RC=5
fi
# A structural observable can be degenerate too: a constant trace has stopped tracking the
# computation (reference varies 0-20). Existing isn't the same as distinguishing (SHARK's first
# gate reproduced perfectly over 98,304 zeros).
if [ "$(cut -d: -f2 "$RES/gate.txt" 2>/dev/null | sort -u | wc -l)" -lt 2 ]; then
    echo "FAILED: the level trace is constant, so it distinguishes nothing" >&2
    RC=6
fi

# Value observable kept beside the gate as evidence, never as the gate itself.
grep -E "^GATE" "$RES/stdout.log" > "$RES/gate_values.txt" 2>/dev/null || true

# Guards that the measured program is the program this row claims (first attempt measured an
# RMSNorm unit test for 14s, exit 0, plausible peak).
NMARK=$(grep -c '^MEM|' "$RES/markers.log" 2>/dev/null || echo 0)
[ "$NMARK" -ge 16 ] || { echo "FAILED: $NMARK of 16 phase markers" >&2; RC=7; }
grep -q '^GPUPEAK|' "$RES/logs/all_markers.txt" 2>/dev/null \
    || { echo "FAILED: no GPUPEAK line, so there is no device peak to publish" >&2; RC=7; }
grep -q '^SEEDED|' "$RES/stderr.log" 2>/dev/null \
    || { echo "FAILED: no SEEDED marker, so the seed did not take" >&2; RC=7; }

# Bank must have been used, not merely requested: helpers silently fall back to host RAM (disk=0
# marker) when MOAI_SPILL_DIR is unwritable, producing a plausible but different step.
if [ "${SPILL:-$SPILL_DEFAULT}" = "1" ]; then
    if grep -q 'disk=0' "$RES/stderr.log" 2>/dev/null; then
        echo "FAILED: the disk bank was requested but a lever reports disk=0, so ciphertexts were" >&2
        echo "        parked in HOST RAM instead. That is a different step; see the LEVER| lines." >&2
        grep -h 'LEVER|.*disk=' "$RES/stderr.log" 2>/dev/null | head -4 >&2
        RC=9
    fi
fi

# Empty device table is this campaign's standing failure shape for a wrong interposition set.
if [ -n "${ATTRIB:-}" ]; then
    if [ -s "$RES/device_table.txt" ] && grep -qv '^#' "$RES/device_table.txt"; then
        head -2 "$RES/device_table.txt"
    else
        echo "FAILED: device channel produced no object rows; check the interposition set" >&2
        RC=8
    fi
fi

snni_finish "$RES" "$T0" "$RC"
exit $RC
