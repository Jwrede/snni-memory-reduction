#!/usr/bin/env python3
"""SIGMA-GPU s4: pinned dealer buffers llamaBuf1, dummyBuf1 1 GiB -> 64 MiB each (measured 275 KiB
and 6.45 MiB, p_llamabuf_highwater). llamaBuf1 needs max(round), dummyBuf1 sum(round). cpuMalloc pins
every page (cudaHostRegister). Saves ~1.875 GiB of 21.05 GiB (~8.9%). Free (oversized). Gate T1s.

    patch_dealerbuf_measured.py <EzPC/GPU-MPC tree>
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

    # dummyBuf2 MUST STILL NEVER BE RESET: the 6,764,544 B figure is a whole-run accumulation, only
    # the right size while that holds. Counts ASSIGNMENTS not mentions (line 302 names it twice).
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
