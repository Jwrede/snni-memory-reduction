#!/usr/bin/env python3
"""SIGMA-GPU s3: keyBufSz 20 -> 18 GiB (bert-base branch). Used 16.83 GiB
(SNNI_BUF|keybuf_used=18075947008); cpuMalloc pins every page. Saves 2 GiB of 23.04 GiB (~8.7%);
6.9% headroom (no bounds check assumed; keySize computed after writing). Free. Gate T1s.

    patch_keybuf_measured.py <EzPC/GPU-MPC tree>
"""
import os
import re
import sys

SRC = "experiments/sigma/sigma.cu"

ANCHOR = """        bw = 50;
        keyBufSz = 20 * OneGB;
"""

PATCH = """        bw = 50;
        // s3_keybuf_measured: 20 GiB were reserved and 16.83 GiB used, measured by this program's
        // own SNNI_BUF marker in s2_commbuf_measured. `cpuMalloc` pins with cudaHostRegister
        // (pin defaults to true), so every reserved page is resident whether written or not: the
        // slack is real memory, not just address space. 18 GiB keeps 6.9% headroom, which is
        // deliberately generous because nothing in this code checks the buffer bound -- keySize is
        // computed as `keyBuf - startPtr` only after all writing is done.
        keyBufSz = 18 * OneGB;
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "s3_keybuf_measured" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the bert-base keyBufSz matched {txt.count(ANCHOR)} times, expected 1")

    # THE BRANCH MUST BE THE MEASURED ONE: keyBufSz is assigned in eight model branches; anchored on
    # `bw = 50;` above so a different model's 20 GB cannot be hit, and bert-base is verified enclosing.
    at = txt.index(ANCHOR)
    head = txt[:at]
    branch = head.rfind('model == "')
    if branch < 0 or 'model == "bert-base"' not in head[branch:branch + 40]:
        sys.exit("FAILED: the matched keyBufSz is not inside the bert-base branch; this step is "
                 "derived from a bert-base measurement and may not resize another model's buffer")

    # THE MEASURED NEED MUST STILL FIT. Guard against a future edit that lowers this further
    # without redoing the measurement: 18 GiB against the 18,075,947,008 B this workload used.
    if 18 * (1 << 30) <= 18075947008:
        sys.exit("FAILED: the new buffer is not larger than the measured need")

    # THE PIN DEFAULT IS WHY THIS SAVES RESIDENT BYTES: if a future tree sets pin=false the unwritten
    # tail stops being resident and the argument collapses while still "working". Checked.
    mem = os.path.join(sys.argv[1], "utils/gpu_mem.h")
    if os.path.exists(mem):
        if not re.search(r"cpuMalloc\s*\(\s*size_t[^)]*bool\s+pin\s*=\s*true", open(mem).read()):
            sys.exit("FAILED: cpuMalloc no longer pins by default, so the reserved-but-unwritten "
                     "tail would not be resident and this step would save nothing")
    else:
        sys.exit("FAILED: utils/gpu_mem.h not found; cannot confirm cpuMalloc still pins")

    txt = txt.replace(ANCHOR, PATCH, 1)
    open(src, "w").write(txt)
    print("patch_keybuf_measured: bert-base key buffer 20 -> 18 GiB (" + SRC + ")")


if __name__ == "__main__":
    main()
