#!/bin/bash
#SBATCH --job-name=sigmabuild
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/j_wred02/sigma_gpu/build-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/sigma_gpu/build-%j.out

# Builds the step binaries. First rebuilds the baseline from source and compares its md5 with the
# binary in use (build record missing): match validates the build path, mismatch flags s0 for
# re-measurement.
set -uo pipefail

B=/scratch/tmp/j_wred02/sigma_gpu
SIF="$B/sigma-gpu-s0_default.sif"
WORK="$B/build_src"
BASELINE_MD5=cd12e14fdaf2e528869ab2a35c841150

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }
[ -r "$B/patches/gate_deterministic_io.diff" ] || {
    echo "FAILED: no gate patch at $B/patches/gate_deterministic_io.diff" >&2; exit 2; }

# Source is copied out of the image, not cloned: the image's tree is what the baseline was
# measured against, and it has no git revision to check out instead.
rm -rf "$WORK"; mkdir -p "$WORK"
apptainer exec "$SIF" tar cf - -C /sigma EzPC 2>/dev/null | tar xf - -C "$WORK" || {
    echo "FAILED: could not copy the source tree out of the image" >&2; exit 2; }
G="$WORK/EzPC/GPU-MPC"
[ -f "$G/utils/gpu_mem.cu" ] || { echo "FAILED: no gpu_mem.cu under $G" >&2; exit 3; }
echo "source tree at $G"

cd "$WORK/EzPC" || exit 2
git init -q . 2>/dev/null
git apply -p1 --check "$B/patches/gate_deterministic_io.diff" 2>&1 || {
    echo "FAILED: the gate patch does not apply to this tree" >&2; exit 3; }
git apply -p1 "$B/patches/gate_deterministic_io.diff" || exit 3
echo "gate patch applied"

# Setup, not a lever: recvBytes asserts a full MSG_WAITALL chunk, but a poller SIGSTOP/SIGCONT
# during a transfer can interrupt it and abort both parties (crashed the _attrib/_pmrec channels
# reproducibly). Fixed identically to LLAMA online's same defect; applied to baseline and every step.
python3 - "$G/utils/sigma_comms.cpp" <<'PYRECV'
import sys
p = sys.argv[1]
s = open(p).read()
old = """        assert(numRead == toRead);
        bytesRead += numRead;"""
if s.count(old) != 1:
    sys.exit("FAILED: the recvBytes assert matched %d times, expected 1" % s.count(old))
new = """        // SNNI setup: a short return is a signal-interrupted MSG_WAITALL, not an error. The
        // loop this sits in already handles it -- `bytesRead` advances and the next iteration
        // asks for the remainder -- so the assert was the only thing turning it into a crash.
        // A genuine close returns 0 and would spin here, so that case is still fatal.
        assert(numRead > 0 && "peer closed mid-transfer");
        bytesRead += numRead;"""
open(p, "w").write(s.replace(old, new))
print("  patched: recvBytes tolerates a signal-interrupted read (setup)")
PYRECV
[ $? -eq 0 ] || { echo "FAILED: the recvBytes setup patch refused" >&2; exit 3; }
grep -q 'peer closed mid-transfer' "$G/utils/sigma_comms.cpp" || {
    echo "FAILED: the recvBytes patch is not in the source" >&2; exit 3; }

# Debug info required: without it, a 32 GiB object (97.2% of host peak) was once published under
# the nearest exported symbol's name (measured 2026-08-06). -g goes to nvcc itself, not only via
# -Xcompiler (the first attempt did that and produced no debug_info); doesn't affect the peak.
sed -i 's/^FLAGS := -O3 /FLAGS := -O3 -g /' "$G/Makefile"
grep -q '^FLAGS := -O3 -g ' "$G/Makefile" || {
    echo "FAILED: could not add -g to FLAGS in $G/Makefile" >&2; exit 3; }
echo "debug info enabled (nvcc -g)"

build() {  # <output name>
    local out="$1"
    # SNNI_BUILD_ONLY=<output name> compiles that one binary and skips every other compile, so
    # adding a step does not overwrite the published binaries of the steps before it. The patches
    # are still applied in order, so the named binary is built from the same cumulative tree.
    if [ -n "${SNNI_BUILD_ONLY:-}" ] && [ "$out" != "$SNNI_BUILD_ONLY" ]; then
        echo "skipped $out (SNNI_BUILD_ONLY=$SNNI_BUILD_ONLY)"; return 0
    fi
    apptainer exec --bind "$WORK":"$WORK" "$SIF" bash -c "cd '$G' && make sigma -j 8" \
        || { echo "FAILED: make sigma failed for $out" >&2; return 1; }
    [ -x "$G/experiments/sigma/sigma" ] || { echo "FAILED: no binary produced for $out" >&2; return 1; }
    mkdir -p "$B/bin"
    cp "$G/experiments/sigma/sigma" "$B/bin/$out"
    # Checked on the artifact (a flag can look present in the Makefile but not reach the
    # compiler), on the host not through the container ($B isn't bound in, so it once falsely
    # rejected two good builds).
    readelf -S "$B/bin/$out" 2>/dev/null | grep -q debug_info || {
        echo "FAILED: $out carries no debug_info; every call site in its object table would be" >&2
        echo "        the nearest exported symbol rather than the allocating function." >&2
        return 1; }
    md5sum "$B/bin/$out"
    rm -f "$G/experiments/sigma/sigma"
}

echo
echo "=== 1. the baseline, rebuilt ==="
build sigma-s0_rebuilt || exit 4
NEW=$(md5sum "$B/bin/sigma-s0_rebuilt" | awk '{print $1}')
echo "rebuilt  md5 = $NEW"
echo "in use   md5 = $BASELINE_MD5"
# A mismatch is now expected (the recvBytes setup patch changes the binary on purpose); what
# matters is baseline and step share one tree and one set of setup patches, both re-measured.
if [ "$NEW" = "$BASELINE_MD5" ]; then
    echo "NOTE: identical to the pre-setup baseline binary -- the recvBytes patch did not take"
    echo "      (it is checked above, so this should be unreachable)"
else
    echo "expected: differs from the pre-setup baseline ($BASELINE_MD5), which carried neither the"
    echo "          recvBytes fix nor any other setup added since. RE-MEASURE s0_default with"
    echo "          sigma-s0_rebuilt before quoting any step against it."
fi

echo
echo "=== 2. s1_nowarmup ==="
python3 - "$G/utils/gpu_mem.cu" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = """    uint64_t *d_dummy_ptr;
    uint64_t bytes = 40 * (1ULL << 30);
    checkCudaErrors(cudaMallocAsync(&d_dummy_ptr, bytes, 0));
    checkCudaErrors(cudaFreeAsync(d_dummy_ptr, 0));
"""
if s.count(old) != 1:
    sys.exit("FAILED: the warmup block matched %d times, expected exactly 1" % s.count(old))
new = """    // s1_nowarmup: the 40 GiB warmup allocation is removed.
    //
    // It allocated 40 GiB and freed it on the next line. Nothing wrote to it, nothing read it, and
    // the pointer never left this function -- its only purpose was to force the pool to RESERVE
    // 40 GiB up front. Because the release threshold above is UINT64_MAX the pool never returned
    // those pages, so the free was not a free and the device held 40 GiB for the whole run.
    //
    // This campaign's own device table names it as 100% of the VRAM peak. Removing it cannot change
    // a value, because a buffer written by nothing and read by nothing carries none. The pool grows
    // on demand instead, which is the runtime cost this step is measured on.
"""
open(p, "w").write(s.replace(old, new))
print("  patched: warmup allocation removed (utils/gpu_mem.cu)")
PYEOF
[ $? -eq 0 ] || { echo "FAILED: the s1 edit refused" >&2; exit 5; }

grep -q "cudaMallocAsync(&d_dummy_ptr" "$G/utils/gpu_mem.cu" && {
    echo "FAILED: the warmup is still in the source after patching" >&2; exit 5; }
build sigma-s1_nowarmup || exit 5

echo
echo "=== 3. p_bufhighwater (PROBE, its peak is never quoted) ==="
# Probe only: 91% of host peak is three hardcoded worst-case constants (keyBufSz, commBufSize x2);
# this prints their true occupancy (values the program already computes) without changing anything.
python3 - "$G/experiments/sigma/sigma.cu" "$G/utils/sigma_comms.cpp" <<'PYEOF'
import sys
sig, comms = sys.argv[1], sys.argv[2]

s = open(sig).read()
old = "    sigma->keySize = sigmaKeygen->keySize;"
if s.count(old) != 1:
    sys.exit("FAILED: the keySize handover matched %d times, expected 1" % s.count(old))
new = old + """
    // p_bufhighwater: what the key buffer actually holds, against what was reserved for it.
    fprintf(stderr, "SNNI_BUF|keybuf_used=%lu|keybuf_alloc=%lu\\n",
            (unsigned long)sigmaKeygen->keySize, (unsigned long)keyBufSz);
    fflush(stderr);"""
open(sig, "w").write(s.replace(old, new, 1))
print("  patched: key buffer high-water reported (experiments/sigma/sigma.cu)")

c = open(comms).read()
old2 = "    assert(memSz < commBufSize);"
if c.count(old2) != 1:
    sys.exit("FAILED: the commBufSize assert matched %d times, expected 1" % c.count(old2))
new2 = """    // p_bufhighwater: remember the largest request ever made of the comm buffer. Reported on
    // every new maximum rather than at exit, because this function has no exit hook and the last
    // line printed is the high-water.
    {
        static size_t snni_max_memsz = 0;
        if (memSz > snni_max_memsz) {
            snni_max_memsz = memSz;
            fprintf(stderr, "SNNI_BUF|commbuf_used=%lu|commbuf_alloc=%lu\\n",
                    (unsigned long)snni_max_memsz, (unsigned long)commBufSize);
            fflush(stderr);
        }
    }
""" + old2
open(comms, "w").write(c.replace(old2, new2, 1))
print("  patched: comm buffer high-water reported (utils/sigma_comms.cpp)")
PYEOF
[ $? -eq 0 ] || { echo "FAILED: the probe edit refused" >&2; exit 6; }
grep -q "SNNI_BUF|keybuf_used" "$G/experiments/sigma/sigma.cu" || {
    echo "FAILED: the key-buffer report is not in the source" >&2; exit 6; }
grep -q "SNNI_BUF|commbuf_used" "$G/utils/sigma_comms.cpp" || {
    echo "FAILED: the comm-buffer report is not in the source" >&2; exit 6; }
build sigma-p_bufhighwater || exit 6

echo
echo "=== 4. s2_commbuf_measured ==="
# Chosen from measurement (p_bufhighwater, job 46106974): key buffer is correctly sized
# (16.83/20 GiB used, skipped), comm buffers hold one 384 KiB message at a time in 5 GiB each, so
# both are shrunk to 64 MiB (170x margin); undersizing would abort via the program's own assert.
python3 - "$G/utils/sigma_comms.h" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = "    size_t commBufSize = 5 * OneGB;"
if s.count(old) != 1:
    sys.exit("FAILED: commBufSize matched %d times, expected exactly 1" % s.count(old))
new = """    // s2_commbuf_measured: 5 GiB stood ready for a message measured at 384 KiB.
    //
    // `p_bufhighwater` recorded the largest request this buffer ever sees (job 46106974): 393,216
    // bytes against 5,368,709,120 reserved, a factor of 13,653, and the class holds TWO of them.
    // The buffer carries ONE message at a time (every use passes `memSz`; gpu_comms.h:215-251),
    // so the reserve serves no accumulation.
    //
    // 64 MiB is 170x the largest message observed. The margin is wide on purpose: the measurement
    // is one run of one pinned workload. Undersizing cannot corrupt a result, because
    // `getMemSz` asserts `memSz < commBufSize` and aborts; oversizing only keeps the memory this
    // step removes.
    size_t commBufSize = 64ULL * 1024 * 1024;"""
open(p, "w").write(s.replace(old, new, 1))
print("  patched: commBufSize 5 GiB -> 64 MiB (utils/sigma_comms.h)")
PYEOF
[ $? -eq 0 ] || { echo "FAILED: the commBufSize edit refused" >&2; exit 7; }
grep -q "commBufSize = 64ULL" "$G/utils/sigma_comms.h" || {
    echo "FAILED: commBufSize is still 5 GiB after patching" >&2; exit 7; }
build sigma-s2_commbuf_measured || exit 7

# --- s3_keybuf_measured -------------------------------------------------------------------
# Key buffer reserves 20 GiB, measured used 16.83 GiB (pinned pages are resident whether written
# or not); shrunk to 18 GiB, leaving margin since nothing in the code checks this bound.
python3 "$B/patches/patch_keybuf_measured.py" "$G" || {
    echo "FAILED: the key buffer edit refused" >&2; exit 8; }
grep -q "keyBufSz = 18 \* OneGB" "$G/experiments/sigma/sigma.cu" || {
    echo "FAILED: keyBufSz is still 20 GiB after patching" >&2; exit 8; }
build sigma-s3_keybuf_measured || exit 8

# --- p_llamabuf_highwater (PROBE) ---------------------------------------------------------
# Probes the two never-measured 1 GiB buffers (llamaBuf, dummyBuf), which are asymmetric: one
# needs max(llamaKeySz), the other sum(llamaKeySz).
python3 "$B/patches/patch_llamabuf_highwater.py" "$G" || {
    echo "FAILED: the llamaBuf high-water probe refused" >&2; exit 9; }
grep -q "snniLlamaHigh" "$G/backend/sigma.h" || {
    echo "FAILED: the high-water tracker is not in the source" >&2; exit 9; }
build sigma-p_llamabuf_highwater || exit 9

# --- s4_dealerbuf_measured ----------------------------------------------------------------
# The probe above measured 275 KiB and 6.45 MiB in two 1 GiB pinned buffers. Both are sized to
# 64 MiB here, the same figure s2 chose for the comm buffer: 238x and 10x margin respectively.
python3 "$B/patches/patch_dealerbuf_measured.py" "$G" || {
    echo "FAILED: the dealer buffer edit refused" >&2; exit 10; }
grep -q "llamaBuf1 = (u8 \*)cpuMalloc(64ULL" "$G/backend/sigma.h" || {
    echo "FAILED: llamaBuf is still 1 GiB after patching" >&2; exit 10; }
build sigma-s4_dealerbuf_measured || exit 10

# --- s5_keybuf_tight ----------------------------------------------------------------------
# Last 1.17 GiB of key-buffer slack: close()'s own assert(keySize < keyBufSize) means an
# over-tight buffer aborts rather than publishing wrong numbers; 17 GiB keeps 169 MB margin.
python3 "$B/patches/patch_keybuf_tight.py" "$G" || {
    echo "FAILED: the tight key buffer edit refused" >&2; exit 11; }
grep -q "keyBufSz = 17 \* OneGB" "$G/experiments/sigma/sigma.cu" || {
    echo "FAILED: keyBufSz is still 18 GiB after patching" >&2; exit 11; }
build sigma-s5_keybuf_tight || exit 11
# Backstop assert must survive into the binary; checked with grep -a, not strings, since strings
# is unavailable on the compute node after module purge (once falsely failed a good binary).
grep -aq "keySize < keyBufSize" "$B/bin/sigma-s5_keybuf_tight" || {
    echo "FAILED: the bound assert is not in the built binary (NDEBUG?), so an over-tight buffer would corrupt silently" >&2; exit 11; }

# --- s6_key_window ------------------------------------------------------------------------
# keyBuf is 94.3% of host peak; a key lives for one operation, so the dealer writes to a file,
# the online phase maps it, and the dealer's buffer releases before the online phase exists.
python3 "$B/patches/patch_key_window.py" "$G" || {
    echo "FAILED: the key window edit refused" >&2; exit 12; }
# Guards that both buffers are never alive at once (the stock program hands the dealer's buffer
# straight to the online phase). Release needs no edit; an explicit cpuFree here was a double
# free that killed job 46262407.
grep -q "sigma->keyBuf = sigmaKeygen->startPtr" "$G/experiments/sigma/sigma.cu" && {
    echo "FAILED: the online phase still inherits the dealer's buffer, so both are alive at once" >&2
    exit 12; }
grep -q "mmap(NULL, keySize" "$G/backend/sigma.h" || {
    echo "FAILED: the online phase still reads the key file into a buffer" >&2; exit 12; }
build sigma-s6_key_window || exit 12

# --- p_keywindow_high (PROBE) -------------------------------------------------------------
# s6 cut the online phase from 18.9 GB to 1.09 GB but moved the peak only 0.71% (VmHWM is
# monotone; the dealer's fill happens first). Probes max bytes appended per operation.
python3 "$B/patches/patch_keywindow_probe.py" "$G" || {
    echo "FAILED: the key-window probe refused" >&2; exit 13; }
grep -q "SNNI_KW|op_high" "$G/backend/sigma.h" || {
    echo "FAILED: the probe's report is not in the source" >&2; exit 13; }
build sigma-p_keywindow_high || exit 13

# --- s6_dealer_window ---------------------------------------------------------------------
# Targets the dealer's fill itself: buffer now holds one operation instead of 97, sized to
# p_keywindow_high's measured largest operation (825,125,000 B) with a 1.30x margin.
python3 "$B/patches/patch_dealer_window.py" "$G" || {
    echo "FAILED: the dealer window edit refused" >&2; exit 14; }
grep -q "keyBufSz = 1 \* OneGB" "$G/experiments/sigma/sigma.cu" || {
    echo "FAILED: the key buffer is not 1 GiB after patching" >&2; exit 14; }
grep -q "kwOut.write" "$G/backend/sigma.h" || {
    echo "FAILED: the rewind does not write, so the key material would be lost" >&2; exit 14; }
build sigma-s6_dealer_window || exit 14
# Bound must survive into the binary (same reason as s5): the margin rests on an over-tight
# window aborting rather than corrupting.
grep -aq "keyBuf - startPtr) < keyBufSize" "$B/bin/sigma-s6_dealer_window" || \
grep -aq "keySize < keyBufSize" "$B/bin/sigma-s6_dealer_window" || {
    echo "FAILED: the window assert is not in the built binary (NDEBUG?)" >&2; exit 14; }

# --- s7_keygen_tight ---------------------------------------------------------------------
# The dealer's three pinned buffers (key window, llamaBuf, dummyBuf) sized to what they measurably
# hold: 800 MiB for a largest operation of 825,125,000 B, 1 MiB for 281,856 B per round, 8 MiB for
# 6,764,544 B in total; each gets an overflow check. Applied to a copy of the two files it edits
# and reverted after the build, so the weight-window stage below still starts from s6's tree.
cp "$G/backend/sigma.h" "$WORK/sigma.h.at_s6"
cp "$G/experiments/sigma/sigma.cu" "$WORK/sigma.cu.at_s6"
python3 "$B/patches/patch_keygen_tight.py" "$G" || {
    echo "FAILED: the keygen buffer edit refused" >&2; exit 16; }
grep -q "keyBufSz = 800ULL \* 1024 \* 1024" "$G/experiments/sigma/sigma.cu" || {
    echo "FAILED: the key window is not 800 MiB after patching" >&2; exit 16; }
grep -q "cpuMalloc(snniLlamaBufSz)" "$G/backend/sigma.h" || {
    echo "FAILED: llamaBuf is not sized by snniLlamaBufSz after patching" >&2; exit 16; }
build sigma-s7_keygen_tight || exit 16
# The margins rest on the three checks aborting, so all three must survive into the binary.
if [ -z "${SNNI_BUILD_ONLY:-}" ] || [ "$SNNI_BUILD_ONLY" = sigma-s7_keygen_tight ]; then
    for m in "overflowed the dealer window" "llamaBuf overflowed" "dummyBuf overflowed"; do
        grep -aq "$m" "$B/bin/sigma-s7_keygen_tight" || {
            echo "FAILED: the check '$m' is not in the built binary (NDEBUG?)" >&2; exit 16; }
    done
fi
# --- s8_weights_dead ----------------------------------------------------------------------
# On top of s7's tree: the online backend releases each layer's FC and attention weight tensors
# with madvise(MADV_DONTNEED) after the synchronous copy that reads them for the last time. The
# same two files are edited, so the s7 restore below reverts both steps.
python3 "$B/patches/patch_weights_dead.py" "$G" || {
    echo "FAILED: the weights-dead edit refused" >&2; exit 17; }
build sigma-s8_weights_dead || exit 17
if [ -z "${SNNI_BUILD_ONLY:-}" ] || [ "$SNNI_BUILD_ONLY" = sigma-s8_weights_dead ]; then
    grep -aq "SNNI_DEAD|" "$B/bin/sigma-s8_weights_dead" || {
        echo "FAILED: the SNNI_DEAD report is not in the built binary" >&2; exit 17; }
fi
cp "$WORK/sigma.h.at_s6" "$G/backend/sigma.h"
cp "$WORK/sigma.cu.at_s6" "$G/experiments/sigma/sigma.cu"
grep -q -e "s7_keygen_tight" -e "s8_weights_dead" "$G/backend/sigma.h" "$G/experiments/sigma/sigma.cu" && {
    echo "FAILED: the s7 edit was not reverted before the weight-window stage" >&2; exit 16; }

# --- s7_weight_window ---------------------------------------------------------------------
# Model weights (32.5% of peak) are dropped via madvise(MADV_DONTNEED) after each layer and
# rewritten before the next (exact, since the fill is a pure function of a counter). Since this
# system's gate is structural, the step carries its own per-tensor checksum, aborting on mismatch.
python3 "$B/patches/patch_weight_window.py" "$G" || {
    echo "FAILED: the weight window edit refused" >&2; exit 15; }
grep -q "snni_weights_in" "$G/ext/sytorch/include/sytorch/layers/layers.h" || {
    echo "FAILED: the per-layer hooks are not in layers.h" >&2; exit 15; }
grep -q "refill mismatch" "$G/experiments/sigma/sigma.cu" || {
    echo "FAILED: the refill checksum is missing, and on this system the gate cannot replace it" >&2
    exit 15; }
build sigma-s8_weight_window_early || exit 15
# THE CHECKSUM MUST SURVIVE INTO THE BINARY. It is the only thing standing between a wrong weight
# and a published number, because the gate is structural.
grep -aq "refill mismatch" "$B/bin/sigma-s8_weight_window_early" || {
    echo "FAILED: the refill check is not in the built binary" >&2; exit 15; }

echo
echo "=== summary ==="
md5sum "$B/bin/sigma-s6_key_window" "$B/bin/sigma-s0_default" "$B/bin/sigma-s0_rebuilt" "$B/bin/sigma-s1_nowarmup" "$B/bin/sigma-p_bufhighwater" "$B/bin/sigma-s2_commbuf_measured" "$B/bin/sigma-s3_keybuf_measured" "$B/bin/sigma-p_llamabuf_highwater" "$B/bin/sigma-s4_dealerbuf_measured" "$B/bin/sigma-s5_keybuf_tight" 2>/dev/null
echo "BUILD_OK"
