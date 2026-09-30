#!/usr/bin/env python3
"""SIGMA-GPU s5: the last 1 GiB of key-buffer slack, backed by the close() assert.

    patch_keybuf_tight.py <EzPC/GPU-MPC tree>

Basis: s4_dealerbuf_measured host peak, rank-1 row (~96%):
    keyBuf      18 GiB   of which 16.83 GiB is written        <- THIS STEP
    llamaBuf1   64 MiB   sized by s4 against measurement
    dummyBuf1   64 MiB   sized by s4 against measurement
1.17 GiB unwritten but resident (cudaHostRegister; s3, s4 predicted their savings to 0.12%, 0.11%).
Backstop: s3's "no bounds check" was wrong. SIGMAKeygen::close() ends with
    assert(keySize < keyBufSize);
live in the binary (NDEBUG not defined); close() runs before the online phase (sigma.cu:
sigmaKeygen->close(), then new SIGMA<u64>(...)). Over-tight buffer: heap corruption, then abort.
Need, deterministic (function of network shape and operation sequence, not PRNG values):
    s2_commbuf_measured    keybuf_used=18075947008
    s3_keybuf_measured     keybuf_used=18075947008
    s4_dealerbuf_measured  keybuf_used=18075947008
Lever: keyBufSz 18 * OneGB -> 17 * OneGB, bert-base branch only.
    17 GiB = 18,253,611,008 B against 18,075,947,008 B needed
    margin 177,664,000 B = 169 MB = 0.98%
    saving 1 GiB = 1,048,576 kB, about 5.0% of the 20.1 GiB peak
Free (oversized). Falsifier: gate T1s must not move; saving far below 1 GiB refutes residency; an
assert abort refutes the need.

Usage: python3 patch_keybuf_tight.py <path to EzPC/GPU-MPC>
"""
import os
import re
import sys

SRC = "experiments/sigma/sigma.cu"
NEED = 18075947008
NEW = 17 * (1 << 30)

ANCHOR = "        keyBufSz = 18 * OneGB;\n"

PATCH = """        // s5_keybuf_tight: 16.83 GiB is needed and was reported byte-identically by three
        // runs of three different binaries; keySize is the sum of per-operation FSS key sizes for
        // a fixed network shape, so it is determined rather than sampled. 17 GiB keeps 169 MB
        // (0.98%). s3 left more because it claimed nothing checked the bound -- close() in fact
        // ends with assert(keySize < keyBufSize), live in the binary and reached before the online
        // phase exists, so an over-tight buffer aborts the run instead of publishing a wrong one.
        keyBufSz = 17 * OneGB;
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "s5_keybuf_tight" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the 18 GiB keyBufSz matched {txt.count(ANCHOR)} times, expected 1; "
                 "this step must be applied on top of s3_keybuf_measured")

    at = txt.index(ANCHOR)
    head = txt[:at]
    branch = head.rfind('model == "')
    if branch < 0 or 'model == "bert-base"' not in head[branch:branch + 40]:
        sys.exit("FAILED: the matched keyBufSz is not inside the bert-base branch")

    if NEW <= NEED:
        sys.exit("FAILED: the new buffer is not larger than the measured need")

    # THE ASSERT IS THE BACKSTOP THIS MARGIN RESTS ON. If a later tree drops it, or defines NDEBUG,
    # an over-tight buffer would corrupt silently again and a 0.98% margin would no longer be
    # defensible. Checked in the source here; the build additionally greps the binary.
    sh = os.path.join(sys.argv[1], "backend/sigma.h")
    if not os.path.exists(sh):
        sys.exit("FAILED: backend/sigma.h not found; cannot confirm the bound is checked")
    if not re.search(r"assert\s*\(\s*keySize\s*<\s*keyBufSize\s*\)", open(sh).read()):
        sys.exit("FAILED: `assert(keySize < keyBufSize)` is gone; this margin was chosen because "
                 "an over-tight buffer aborts rather than corrupting silently, and without that "
                 "backstop 0.98% is not defensible")

    # PINNING IS STILL WHY THIS SAVES RESIDENT BYTES.
    mem = os.path.join(sys.argv[1], "utils/gpu_mem.h")
    if not os.path.exists(mem) or not re.search(
            r"cpuMalloc\s*\(\s*size_t[^)]*bool\s+pin\s*=\s*true", open(mem).read()):
        sys.exit("FAILED: cpuMalloc no longer pins by default, so the unwritten tail would not be "
                 "resident and this step would save nothing")

    txt = txt.replace(ANCHOR, PATCH, 1)
    open(src, "w").write(txt)
    print("patch_keybuf_tight: bert-base key buffer 18 -> 17 GiB (" + SRC + ")")


if __name__ == "__main__":
    main()
