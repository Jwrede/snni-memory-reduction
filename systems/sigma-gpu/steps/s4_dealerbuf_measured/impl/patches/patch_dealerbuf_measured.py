#!/usr/bin/env python3
"""SIGMA-GPU s4: two 1 GiB pinned dealer buffers holding 275 KiB and 6.45 MiB.

    patch_dealerbuf_measured.py <EzPC/GPU-MPC tree>

Basis: s3_keybuf_measured host peak:
    21474.8 MB  95.0%  heap:sigma.cu:248:main  <- gpu_mem.cu:77:cpuMalloc   <- THIS STEP
      547.6 MB   2.4%  heap:bert.h:89:GPUBERT<unsigned long>::GPUBERT
Row = 20 GiB, three pinned allocations: keyBuf 18 GiB (s3), llamaBuf1 1 GiB, dummyBuf1 1 GiB.
Need (p_llamabuf_highwater, job 46142898):
    SNNI_BUF|llamabuf_high=281856|llamabuf_alloc=1073741824      275 KiB of 1 GiB, factor 3810
    SNNI_BUF|dummybuf_used=6764544|dummybuf_alloc=1073741824    6.45 MiB of 1 GiB, factor  159
Asymmetry: initDealer (backend/sigma.h:302) hands one party llamaBuf2, the other dummyBuf2:
    size_t llamaKeySz = llamaBuf2 - llamaBuf1;
    memcpy(keyBuf, llamaBuf1, llamaKeySz);
    llamaBuf2 = llamaBuf1;                       // reset -- holds ONE round
dummyBuf2 assigned once, never reset: accumulates. llamaBuf1 needs max(llamaKeySz), dummyBuf1
sum(llamaKeySz).
Lever: both cpuMalloc(OneGB) in the SIGMAKeygen constructor -> 64 MiB (238x the largest round, 10x
the whole-run sum; same figure as s2's comm buffer).
    2 GiB -> 128 MiB, i.e. 1.875 GiB off a 21.05 GiB peak, about 8.9%
Margin wide: offsets by pointer subtraction after writing (no per-buffer bounds check); 16 MiB
would add 0.3%.
Donor: BumbleBee b4_slice_budget_64m; this line's s2, s3. Free (oversized).
Falsifier: gate T1s must not move; a drop far below 1.875 GiB refutes the pinning argument.

Usage: python3 patch_dealerbuf_measured.py <path to EzPC/GPU-MPC>
"""
import os
import re
import sys

SRC = "backend/sigma.h"

ANCHOR = """        llamaBuf1 = (u8 *)cpuMalloc(OneGB);
        dummyBuf1 = (u8 *)cpuMalloc(OneGB);
"""

PATCH = """        // s4_dealerbuf_measured: 1 GiB each held 275 KiB and 6.45 MiB, measured by
        // p_llamabuf_highwater (job 46142898). Both are pinned by cudaHostRegister, so every
        // reserved page was resident. llamaBuf is drained and reset per round and needs the
        // largest single round; dummyBuf is never reset and needs the whole-run accumulation --
        // which is why both were measured rather than one argued from the other. 64 MiB leaves
        // 238x and 10x, and matches the figure s2_commbuf_measured chose for the comm buffer.
        llamaBuf1 = (u8 *)cpuMalloc(64ULL * 1024 * 1024);
        dummyBuf1 = (u8 *)cpuMalloc(64ULL * 1024 * 1024);
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "s4_dealerbuf_measured" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the two dealer allocations matched {txt.count(ANCHOR)} times, "
                 "expected exactly 1")

    # THE MEASURED NEEDS MUST STILL FIT, checked against the probe's own numbers so that a later
    # edit cannot shrink these further without redoing the measurement.
    NEW = 64 * 1024 * 1024
    if NEW <= 281856:
        sys.exit("FAILED: the new llamaBuf is not larger than the measured 281,856 B high-water")
    if NEW <= 6764544:
        sys.exit("FAILED: the new dummyBuf is not larger than the measured 6,764,544 B used")

    # dummyBuf2 MUST STILL NEVER BE RESET. The 6,764,544 B figure is a whole-run accumulation and
    # is only the right number to size against while that holds. If a later tree resets it, the
    # buffer would need only one round and this margin would be wrong in the other direction --
    # harmless, but the derivation would no longer describe the code. Counting ASSIGNMENTS, not
    # mentions: line 302 names `dummyBuf2` twice, once per ternary.
    assigns = len(re.findall(r"dummyBuf2\s*=(?!=)", txt))
    if assigns != 1:
        sys.exit(f"FAILED: `dummyBuf2` is assigned {assigns} times, expected exactly 1; the "
                 "measured accumulation no longer describes this code")

    # PINNING IS THE REASON THIS SAVES RESIDENT BYTES AT ALL.
    mem = os.path.join(sys.argv[1], "utils/gpu_mem.h")
    if not os.path.exists(mem):
        sys.exit("FAILED: utils/gpu_mem.h not found; cannot confirm cpuMalloc still pins")
    if not re.search(r"cpuMalloc\s*\(\s*size_t[^)]*bool\s+pin\s*=\s*true", open(mem).read()):
        sys.exit("FAILED: cpuMalloc no longer pins by default, so the reserved-but-unwritten tail "
                 "would not be resident and this step would save nothing")

    txt = txt.replace(ANCHOR, PATCH, 1)
    open(src, "w").write(txt)
    print("patch_dealerbuf_measured: llamaBuf/dummyBuf 1 GiB -> 64 MiB each (" + SRC + ")")


if __name__ == "__main__":
    main()
