#!/usr/bin/env python3
"""p_keywindow_high probe: key bytes produced by one dealer operation. Changes nothing.

Origin: s6_key_window moved the online phase from 18.9 GB to 1.09 GB; published peak -0.71%:
    t = 0.0 s   VmRSS 18,916,808 kB      the dealer's key buffer, already full
    t = 3.5 s   VmRSS  1,091,092 kB      close() writes the file and frees it
    t = 3.5 to 63 s   VmRSS stays at ~1.1 to 1.9 GB, VmHWM frozen at 18,916,820
Peak = dealer fill, before the online phase (cf. MOAI-GPU before_attention, SHAFT-CPU s7_embed_chunk).
Next step: rewind after every operation (all eight keyBuf sites append; startPtr read only for
keySize and the final write); buffer = max over operations of bytes appended.
Output at the dealer's close, per party:
    SNNI_KW|op_high=<bytes>|ops=<count>|keysize=<bytes>
op_high = window a rewinding dealer needs; keysize = current need.
kwNote() compares two pointers and assigns one; key material, order and transcript unchanged.
"""

import os
import sys

H = "backend/sigma.h"

# ---- the counter, in the KEYGEN class only ----
# `    u8 *startPtr;` without an initialiser is the keygen's; the online class writes
# `    u8 *startPtr = NULL;`. The three-line anchor makes that unambiguous.
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

    # EVERY SITE MUST BE INSTRUMENTED. Eight is the number of methods that advance the cursor; if a
    # future tree grows a ninth, `op_high` would silently under-report and the next step would size
    # its buffer too small, which is an abort rather than a wrong number but still a wasted run.
    if txt.count("kwNote();") != len(SITES):
        sys.exit(f"FAILED: {txt.count('kwNote();')} call sites written, expected {len(SITES)}")

    open(hp, "w").write(txt)
    print(f"patch_keywindow_probe: {len(SITES)} append sites instrumented, report at the dealer's "
          f"close, nothing else changed")


if __name__ == "__main__":
    main()
