#!/usr/bin/env python3
"""s8_weights_dead: each layer's weight tensors released after their last online read.

One inference; in the online phase a layer's host weights are read once, by the synchronous
cudaMemcpy in moveToGPU (gpuMatmul / gpuMHA), then never (no later layer, sigma->output or gate).
madvise(MADV_DONTNEED) on the region rounded inward to whole pages. No value rewritten, nothing
recomputed: exact and free, 96 madvise calls per party.
Online backend only (class SIGMA, flag set in sigma.cu); the dealer (earlier reads) is unchanged.
"""

import os
import sys

H = "backend/sigma.h"
CU = "experiments/sigma/sigma.cu"

MEMBER_OLD = """    u8 *kwBase = NULL;
    size_t kwLen = 0, kwDropped = 0;
"""
MEMBER_NEW = """    u8 *kwBase = NULL;
    size_t kwLen = 0, kwDropped = 0;

    // s8_weights_dead: a layer's host weights are dead once this backend has copied them to the
    // device, because the program runs one inference and nothing reads them afterwards. Set by
    // sigma.cu for the online phase only; off by default.
    bool snniDropWeights = false;
    u64 snniDeadDrops = 0, snniDeadBytes = 0;
    void snniDropDead(const void *p, size_t bytes)
    {
        if (!snniDropWeights || p == NULL)
            return;
        const uintptr_t PG = 4096;
        uintptr_t a = (uintptr_t)p, b = a + bytes;
        uintptr_t lo = (a + PG - 1) & ~(PG - 1), hi = b & ~(PG - 1);
        if (hi <= lo)
            return;
        if (madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED) == 0)
        {
            ++snniDeadDrops;
            snniDeadBytes += (u64)(hi - lo);
        }
    }
"""

MATMUL_OLD = """        c.d_data = gpuMatmul(peer, party, p, k, a.d_data, b.data, useBias ? d.data : (T *)NULL, TruncateType::None, &g, &s, false);
"""
MATMUL_NEW = MATMUL_OLD + """        // s8_weights_dead: `b` (the FC weight) and `d` (its bias) were read by the synchronous copy
        // above for the last time.
        snniDropDead(b.data, b.d1 * b.d2 * sizeof(T));
        if (useBias)
            snniDropDead(d.data, d.d1 * sizeof(T));
"""

MHA_OLD = """        Y.d_data = gpuMHA(peer, party, bw, scale, pMHA, pMHAMul, k, wQKV.data, bQKV.data, wProj.data, bProj.data, X.d_data, d_mhaTab, &g, &s);
"""
MHA_NEW = MHA_OLD + """        // s8_weights_dead: the four attention parameter tensors were read for the last time above.
        snniDropDead(wQKV.data, wQKV.d1 * wQKV.d2 * sizeof(T));
        snniDropDead(wProj.data, wProj.d1 * wProj.d2 * sizeof(T));
        snniDropDead(bQKV.data, bQKV.d1 * sizeof(T));
        snniDropDead(bProj.data, bProj.d1 * sizeof(T));
"""

ARM_OLD = """    net->setBackend(sigma);
"""
ARM_NEW = """    net->setBackend(sigma);
    // s8_weights_dead: the online backend releases each layer's weights after their last read.
    sigma->snniDropWeights = true;
"""

REPORT_OLD = """    sigma->close();
"""
REPORT_NEW = """    sigma->close();
    fprintf(stderr, "SNNI_DEAD|drops=%lu|bytes=%lu\\n", (unsigned long)sigma->snniDeadDrops,
            (unsigned long)sigma->snniDeadBytes);
    fflush(stderr);
"""


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: patch_weights_dead.py <source tree>")
    tree = sys.argv[1]
    hp, cup = os.path.join(tree, H), os.path.join(tree, CU)
    for f in (hp, cup):
        if not os.path.exists(f):
            sys.exit(f"FAILED: {f} does not exist")
    h, cu = open(hp).read(), open(cup).read()
    if "snniDropWeights" in h or "snniDropWeights" in cu:
        sys.exit("FAILED: this tree already carries the lever")
    # madvise is already used by the s6 key window; the lever relies on that include.
    if "#include <sys/mman.h>" not in h:
        sys.exit("FAILED: backend/sigma.h does not include <sys/mman.h>; is the s6 tree in place?")
    for name, old, txt in (("member", MEMBER_OLD, h), ("online matmul", MATMUL_OLD, h),
                           ("online mha", MHA_OLD, h), ("arm point", ARM_OLD, cu),
                           ("report point", REPORT_OLD, cu)):
        if txt.count(old) != 1:
            sys.exit(f"FAILED: anchor `{name}` matched {txt.count(old)} times, expected exactly 1")
    h = h.replace(MEMBER_OLD, MEMBER_NEW, 1).replace(MATMUL_OLD, MATMUL_NEW, 1) \
         .replace(MHA_OLD, MHA_NEW, 1)
    cu = cu.replace(ARM_OLD, ARM_NEW, 1).replace(REPORT_OLD, REPORT_NEW, 1)
    open(hp, "w").write(h)
    open(cup, "w").write(cu)
    print("patch_weights_dead: online backend releases FC and attention weights after their last "
          "read; SNNI_DEAD report after sigma->close()")


if __name__ == "__main__":
    main()
