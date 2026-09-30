#!/usr/bin/env python3
"""SIGMA-GPU s3: key buffer reserves 20 GiB, uses 16.83 GiB, every page pinned.

    patch_keybuf_measured.py <EzPC/GPU-MPC tree>

Basis: s2_commbuf_measured host peak:
    23622.3 MB  95.5%  heap:sigma.cu:242:main  <- gpu_mem.cu:77:cpuMalloc   <- THIS STEP
      547.6 MB   2.2%  bert.h:89:GPUBERT<unsigned long>::GPUBERT
       75.5 MB   0.3%  file:nvidiactl
Row = new SIGMAKeygen<u64>(party, bw, scale, "", keyBufSz), three pinned cpuMalloc allocations:
    keyBuf      keyBufSz = 20 * OneGB   21474.8 MB
    llamaBuf1              1 * OneGB     1073.7 MB
    dummyBuf1              1 * OneGB     1073.7 MB
                                        23622.3 MB   <- the published row, to the byte
Residency: cpuMalloc(size_t, bool pin = true) = posix_memalign + cudaHostRegister;
backend/sigma.h:289 passes one argument: all 20 GiB page-locked.
Need: SNNI_BUF|keybuf_used=18075947008|keybuf_alloc=21474836480 (16.83 of 20.00 GiB, 84.2%;
steps/s2_commbuf_measured/attrib/resident/run1/party0.log): 3.17 GiB pinned, never written.
Lever: keyBufSz 20 * OneGB -> 18 * OneGB, bert-base branch only (model=bert-base n_seq=128 n_layer=12
n_head=12 n_embd=768 bw=50 scale=12). Saves 2 GiB of 23.04 GiB (~8.7%).
Margin 6.9%: keySize = keyBuf - startPtr computed after writing (backend/sigma.h:307), no bounds
check assumed (corrected 2026-08-15 in levers.env: close() asserts keySize < keyBufSize).
Donor: BumbleBee b4_slice_budget_64m (hardcoded budget corrected against the run).
Free (oversized). Falsifier: gate T1s must not move; a crash means the buffer is too small.

Usage: python3 patch_keybuf_measured.py <path to EzPC/GPU-MPC>
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

    # THE BRANCH MUST BE THE MEASURED ONE. `keyBufSz` is assigned in eight model branches
    # (1, 10*mul, 20, 50, 80, 200, 300, 450 GB) and only bert-base is what this campaign runs.
    # Anchored on `bw = 50;` immediately above precisely so a different model's 20 GB cannot be
    # hit by accident; checked here that bert-base really is the enclosing branch.
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

    # THE PIN DEFAULT IS THE REASON THIS SAVES RESIDENT BYTES. If a future tree changes it to
    # `pin = false`, the unwritten tail stops being resident, the step's whole argument collapses,
    # and it would still "work" while measuring nothing. Checked rather than assumed.
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
