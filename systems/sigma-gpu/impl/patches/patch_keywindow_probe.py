#!/usr/bin/env python3
"""p_keywindow_high probe: key bytes produced by one dealer operation (kwNote compares two pointers;
no change). Context: s6_key_window showed the peak is the dealer's fill. All keyBuf sites append, so
a rewinding dealer needs max-over-operations bytes (op_high); keysize = current need.

    SNNI_KW|op_high=<bytes>|ops=<count>|keysize=<bytes>
"""

import os
import sys

H = "backend/sigma.h"

# ---- the counter, in the KEYGEN class only ----
# Anchored on the keygen's uninitialised `u8 *startPtr;` (the online class writes `= NULL`), a
# three-line anchor to disambiguate.
FIELDS_OLD = '''    u8 *startPtr;
    u8 *keyBuf = NULL;
    size_t keyBufSize = 0;
'''
FIELDS_NEW = '''    u8 *startPtr;
    u8 *keyBuf = NULL;
    size_t keyBufSize = 0;

    // p_keywindow_high: the largest number of bytes ONE operation appends to keyBuf. That is the
    // buffer a dealer that wrote and rewound after every operation would need.
    size_t snniKwHigh = 0, snniKwOps = 0;
    u8 *kwMark = NULL;
    void kwNote()
    {
        size_t d = (size_t)(keyBuf - kwMark);
        if (d > snniKwHigh) snniKwHigh = d;
        kwMark = keyBuf;
        snniKwOps++;
    }
'''

INIT_OLD = '''        keyBuf = cpuMalloc(keyBufSize);
        startPtr = keyBuf;
'''
INIT_NEW = '''        keyBuf = cpuMalloc(keyBufSize);
        startPtr = keyBuf;
        kwMark = keyBuf;
'''

# ---- the eight append sites, each closed with kwNote() ----
SITES = [
    ("matmul",
     '''        c.d_data = gpuKeygenMatmul<T>(&keyBuf, party, p, a.d_data, b.data, (T *)NULL, TruncateType::None, &g, false);
''',
     '''        c.d_data = gpuKeygenMatmul<T>(&keyBuf, party, p, a.d_data, b.data, (T *)NULL, TruncateType::None, &g, false);
        kwNote();
'''),
    ("gelu",
     '''        out.d_data = gpuKeyGenGelu<T, u8, 8>(&keyBuf, party, bw, bw - scale, (int)scale, in.size(), in.d_data, &g);
''',
     '''        out.d_data = gpuKeyGenGelu<T, u8, 8>(&keyBuf, party, bw, bw - scale, (int)scale, in.size(), in.d_data, &g);
        kwNote();
'''),
    ("silu",
     '''        out.d_data = gpuKeyGenGelu<T, u16, 10>(&keyBuf, party, bw, bw - scale, (int)scale, in.size(), in.d_data, &g);
''',
     '''        out.d_data = gpuKeyGenGelu<T, u16, 10>(&keyBuf, party, bw, bw - scale, (int)scale, in.size(), in.d_data, &g);
        kwNote();
'''),
    ("layernorm",
     '''        keyBuf += llamaKeySz;
        llamaBuf2 = llamaBuf1;
''',
     '''        keyBuf += llamaKeySz;
        kwNote();
        llamaBuf2 = llamaBuf1;
'''),
    ("mha",
     '''        Y.d_data = gpuKeygenMHA(&keyBuf, party, bw, scale, pMHA, pMHAMul, wQKV.data, bQKV.data, wProj.data, bProj.data, X.d_data, &g);
''',
     '''        Y.d_data = gpuKeygenMHA(&keyBuf, party, bw, scale, pMHA, pMHAMul, wQKV.data, bQKV.data, wProj.data, bProj.data, X.d_data, &g);
        kwNote();
'''),
    ("mul",
     '''        out.d_data = gpuKeygenMul(&keyBuf, party, bw, scale, a.size(), a.d_data, b.d_data, TruncateType::None, &g);
''',
     '''        out.d_data = gpuKeygenMul(&keyBuf, party, bw, scale, a.size(), a.d_data, b.d_data, TruncateType::None, &g);
        kwNote();
'''),
    ("truncate",
     '''        in.d_data = genGPUTruncateKey<T, T>(&keyBuf, party, t, bw, bw, shift, in.size(), in.d_data, &g);
''',
     '''        in.d_data = genGPUTruncateKey<T, T>(&keyBuf, party, t, bw, bw, shift, in.size(), in.d_data, &g);
        kwNote();
'''),
    ("output",
     '''        moveIntoCPUMem((u8 *)keyBuf, (u8 *)a.d_data, memSz, (Stats *)NULL);
        keyBuf += memSz;
''',
     '''        moveIntoCPUMem((u8 *)keyBuf, (u8 *)a.d_data, memSz, (Stats *)NULL);
        keyBuf += memSz;
        kwNote();
'''),
]

# ---- the report, next to the two SNNI_BUF lines close() already prints ----
REPORT_OLD = '''        fflush(stderr);
        /*size_t*/ keySize = keyBuf - startPtr;
'''
REPORT_NEW = '''        fprintf(stderr, "SNNI_KW|op_high=%lu|ops=%lu|keysize=%lu\\n",
                (unsigned long)snniKwHigh, (unsigned long)snniKwOps,
                (unsigned long)(keyBuf - startPtr));
        fflush(stderr);
        /*size_t*/ keySize = keyBuf - startPtr;
'''


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: patch_keywindow_probe.py <source tree>")
    tree = sys.argv[1]
    hp = os.path.join(tree, H)
    if not os.path.exists(hp):
        sys.exit(f"FAILED: {hp} does not exist")
    txt = open(hp).read()

    if "snniKwHigh" in txt:
        sys.exit("FAILED: this tree already carries the probe")

    anchors = [("counter fields", FIELDS_OLD), ("buffer init", INIT_OLD),
               ("close report", REPORT_OLD)] + [(n, o) for n, o, _ in SITES]
    for name, old in anchors:
        if txt.count(old) != 1:
            sys.exit(f"FAILED: anchor `{name}` matched {txt.count(old)} times, expected exactly 1")

    txt = txt.replace(FIELDS_OLD, FIELDS_NEW, 1)
    txt = txt.replace(INIT_OLD, INIT_NEW, 1)
    txt = txt.replace(REPORT_OLD, REPORT_NEW, 1)
    for _, old, new in SITES:
        txt = txt.replace(old, new, 1)

    # EVERY SITE MUST BE INSTRUMENTED: eight is the number of cursor-advancing methods; a future ninth
    # would make op_high under-report and size the next buffer too small (an abort, a wasted run).
    if txt.count("kwNote();") != len(SITES):
        sys.exit(f"FAILED: {txt.count('kwNote();')} call sites written, expected {len(SITES)}")

    open(hp, "w").write(txt)
    print(f"patch_keywindow_probe: {len(SITES)} append sites instrumented, report at the dealer's "
          f"close, nothing else changed")


if __name__ == "__main__":
    main()
