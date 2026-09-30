#!/usr/bin/env python3
"""s7_keygen_tight: the dealer constructor's three pinned buffers (sigma.h:371 key window,
sigma.h:392/393 llamaBuf, dummyBuf) sized to measured need, each with an abort-on-overflow check.

    key window  1 GiB  -> 800 MiB   need 825,125,000 B per operation (SNNI_KW|op_high, s6 and
                                    p_keywindow_high, both parties, byte-identical)
    llamaBuf   64 MiB  ->   1 MiB   need   281,856 B per round   (SNNI_BUF|llamabuf_high)
    dummyBuf   64 MiB  ->   8 MiB   need 6,764,544 B accumulated (SNNI_BUF|dummybuf_used)

All cpuMalloc(pin=true): every reserved page resident. Same kernels, bytes, order: gate (COMM/KEYS)
expected identical. New check in kwNote() after every appending operation (s6's close() assert
bounded only the tail after the last rewind). A check fires after the overflow write; the run aborts.

    patch_keygen_tight.py <sigma source tree>
"""

import os
import sys

CU = "experiments/sigma/sigma.cu"
H = "backend/sigma.h"

# ---- the window: bert-base branch only (two-line anchor, as patch_dealer_window.py) ----
BUF_OLD = """        keyBufSz = 1 * OneGB;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);"""
BUF_NEW = """        // s7_keygen_tight: 825,125,000 B is the largest single operation (SNNI_KW|op_high,
        // byte-identical in p_keywindow_high and in s6_dealer_window on both parties). Key sizes
        // are a function of the network shape, so the figure is determined rather than sampled.
        // 800 MiB = 838,860,800 B keeps 13.7 MB (1.7%); kwNote() aborts on overflow.
        keyBufSz = 800ULL * 1024 * 1024;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);"""

# ---- the per-operation bound on the window ----
KW_OLD = """    void kwNote()
    {
        size_t d = (size_t)(keyBuf - kwMark);"""
KW_NEW = """    void kwNote()
    {
        // s7_keygen_tight: one operation must fit the window. Checked after every appending
        // operation; the assert at the end of close() sees only the tail.
        assert((size_t)(keyBuf - startPtr) < keyBufSize && "s7: one operation overflowed the dealer window");
        size_t d = (size_t)(keyBuf - kwMark);"""

# ---- the two LLAMA buffers: sizes named, so the checks and the report use the same figure ----
MEMBER_OLD = """    size_t snniLlamaHigh = 0;
"""
MEMBER_NEW = """    size_t snniLlamaHigh = 0;
    // s7_keygen_tight: measured need 281,856 B per round (llamaBuf, reset every round) and
    // 6,764,544 B in total (dummyBuf, never reset). 1 MiB and 8 MiB; both checked every round.
    size_t snniLlamaBufSz = 1ULL * 1024 * 1024;
    size_t snniDummyBufSz = 8ULL * 1024 * 1024;
"""

ALLOC_OLD = """        llamaBuf1 = (u8 *)cpuMalloc(64ULL * 1024 * 1024);
        dummyBuf1 = (u8 *)cpuMalloc(64ULL * 1024 * 1024);"""
ALLOC_NEW = """        llamaBuf1 = (u8 *)cpuMalloc(snniLlamaBufSz);   // s7_keygen_tight
        dummyBuf1 = (u8 *)cpuMalloc(snniDummyBufSz);   // s7_keygen_tight"""

ROUND_OLD = """        size_t llamaKeySz = llamaBuf2 - llamaBuf1;
        if (llamaKeySz > snniLlamaHigh) snniLlamaHigh = llamaKeySz;"""
ROUND_NEW = """        size_t llamaKeySz = llamaBuf2 - llamaBuf1;
        // s7_keygen_tight: both LLAMA buffers must hold what this round wrote.
        assert(llamaKeySz < snniLlamaBufSz && "s7: llamaBuf overflowed");
        assert((size_t)(dummyBuf2 - dummyBuf1) < snniDummyBufSz && "s7: dummyBuf overflowed");
        if (llamaKeySz > snniLlamaHigh) snniLlamaHigh = llamaKeySz;"""

# ---- the probe's report states the allocation it now measures against ----
REPORT_OLD = """        fprintf(stderr, "SNNI_BUF|llamabuf_high=%lu|llamabuf_alloc=%lu\\n",
                (unsigned long)snniLlamaHigh, (unsigned long)(1024UL * 1024 * 1024));
        fprintf(stderr, "SNNI_BUF|dummybuf_used=%lu|dummybuf_alloc=%lu\\n",
                (unsigned long)(dummyBuf2 - dummyBuf1), (unsigned long)(1024UL * 1024 * 1024));"""
REPORT_NEW = """        fprintf(stderr, "SNNI_BUF|llamabuf_high=%lu|llamabuf_alloc=%lu\\n",
                (unsigned long)snniLlamaHigh, (unsigned long)snniLlamaBufSz);
        fprintf(stderr, "SNNI_BUF|dummybuf_used=%lu|dummybuf_alloc=%lu\\n",
                (unsigned long)(dummyBuf2 - dummyBuf1), (unsigned long)snniDummyBufSz);"""


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: patch_keygen_tight.py <source tree>")
    tree = sys.argv[1]
    cup, hp = os.path.join(tree, CU), os.path.join(tree, H)
    for f in (cup, hp):
        if not os.path.exists(f):
            sys.exit(f"FAILED: {f} does not exist")
    cu, h = open(cup).read(), open(hp).read()

    if "s7_keygen_tight" in cu or "s7_keygen_tight" in h:
        sys.exit("FAILED: this tree already carries the lever")
    # The window must already rewind per operation (s6_dealer_window), or 800 MiB cannot hold the
    # 16.83 GiB stream and the first check aborts.
    if "kwOut.write" not in h:
        sys.exit("FAILED: the dealer does not rewind (patch_dealer_window missing), so an 800 MiB "
                 "window would have to hold the whole key stream")
    # The weight-window lever is on no published row and must not be in this binary.
    if "snni_weights_in" in cu:
        sys.exit("FAILED: the tree carries patch_weight_window, which is on no published row")

    for name, old, txt in (("key window size", BUF_OLD, cu),
                           ("kwNote head", KW_OLD, h),
                           ("llama high-water member", MEMBER_OLD, h),
                           ("llama buffer allocation", ALLOC_OLD, h),
                           ("layernorm round", ROUND_OLD, h),
                           ("close() report", REPORT_OLD, h)):
        if txt.count(old) != 1:
            sys.exit(f"FAILED: anchor `{name}` matched {txt.count(old)} times, expected exactly 1")

    cu = cu.replace(BUF_OLD, BUF_NEW, 1)
    for old, new in ((KW_OLD, KW_NEW), (MEMBER_OLD, MEMBER_NEW), (ALLOC_OLD, ALLOC_NEW),
                     (ROUND_OLD, ROUND_NEW), (REPORT_OLD, REPORT_NEW)):
        h = h.replace(old, new, 1)

    open(cup, "w").write(cu)
    open(hp, "w").write(h)
    print("patch_keygen_tight: window 1 GiB -> 800 MiB, llamaBuf 64 -> 1 MiB, dummyBuf 64 -> 8 MiB, "
          "three overflow checks")


if __name__ == "__main__":
    main()
