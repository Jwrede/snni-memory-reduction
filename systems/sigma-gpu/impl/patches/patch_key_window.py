#!/usr/bin/env python3
"""SIGMA-GPU s6: key material mapped and consumed through a window (mmap + MADV_DONTNEED behind the
cursor); dealer buffer released before the online phase is constructed. Reader only advances a
cursor; a key lives one operation. Paid (16.83 GiB written and read once). Gate expected identical.
Host peak expected ~19,052,324 -> 1.5 to 2.5 M kB.

    patch_key_window.py <EzPC/GPU-MPC tree>
"""

import os
import re
import sys

CU = "experiments/sigma/sigma.cu"
H = "backend/sigma.h"

# ---- sigma.cu: the dealer writes, the online phase maps, the dealer's buffer goes ----
KEYGEN_OLD = '''    auto sigmaKeygen = new SIGMAKeygen<u64>(party, bw, scale, "", keyBufSz);
'''
KEYGEN_NEW = '''    // s6_key_window: the dealer WRITES its key file. The path already exists in `close()` and is
    // disabled only by the empty string this call used to pass.
    auto keyPath = inferenceDir + "key";
    auto sigmaKeygen = new SIGMAKeygen<u64>(party, bw, scale, keyPath, keyBufSz);
'''

ONLINE_OLD = '''    auto sigma = new SIGMA<u64>(party, ip, "", bw, scale, n_seq, n_embd, atoi(__argv[5]), false);
    sigma->keyBuf = sigmaKeygen->startPtr;
    sigma->startPtr = sigma->keyBuf;
    sigma->keySize = sigmaKeygen->keySize;
'''
ONLINE_NEW = '''    // s6_key_window: THE DEALER'S BUFFER IS ALREADY GONE HERE, and it is gone BECAUSE this step
    // writes a key file. `close()` ends with `if (keyFile != "") { write; cpuFree(startPtr); }`, and
    // `cpuFree`'s `pinned` parameter defaults to TRUE, so the file-writing branch both unregisters
    // and frees. In the stock configuration that branch is dead (empty string), which is why the
    // stock program then hands the SAME buffer to the online phase in the three lines this hunk
    // replaces. Enabling the file therefore removes the 17 GiB by itself.
    //
    // AN EXPLICIT `cpuFree(sigmaKeygen->startPtr, true)` STOOD HERE AND WAS A DOUBLE FREE. Measured,
    // job 46262407: both parties died in 41 s at `gpu_mem.cu:91` with code 713
    // `cudaErrorHostMemoryNotRegistered`, immediately after `close()` printed its SNNI_BUF lines.
    // The pointer had already been unregistered and freed one call earlier.
    sigmaKeygen->startPtr = NULL;
    sigmaKeygen->keyBuf = NULL;
    // The online phase MAPS the file rather than inheriting the buffer.
    auto sigma = new SIGMA<u64>(party, ip, keyPath, bw, scale, n_seq, n_embd, atoi(__argv[5]), false);
'''

# ---- backend/sigma.h: map instead of read ----
READ_OLD = '''            getAlignedBuf(&keyBuf, keySize);
            readKey(fd, keySize, keyBuf, NULL);
            startPtr = keyBuf;
'''
READ_NEW = '''            // s6_key_window: map the key file instead of allocating keySize and reading it in.
            // MAP_PRIVATE with PROT_WRITE so a reader that writes gets a private copy rather than a
            // fault; the cursor never revisits a byte, so those copies are never read back.
            void *_kw_map = mmap(NULL, keySize, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
            if (_kw_map == MAP_FAILED)
            {
                perror("s6_key_window: mmap of the key file");
                exit(1);
            }
            madvise(_kw_map, keySize, MADV_SEQUENTIAL);
            keyBuf = (u8 *)_kw_map;
            startPtr = keyBuf;
            kwBase = keyBuf;
            kwLen = keySize;
            kwDropped = 0;
'''

FIELDS_OLD = '''    u8 *startPtr = NULL;
    u8 *keyBuf = NULL;
'''
FIELDS_NEW = '''    u8 *startPtr = NULL;
    u8 *keyBuf = NULL;

    // s6_key_window: the mapping, and how much of it has already been handed back to the kernel.
    u8 *kwBase = NULL;
    size_t kwLen = 0, kwDropped = 0;

    // Drop the pages the cursor has passed. Called immediately BEFORE each key is read, which is
    // the one moment when the previous key is provably dead (it left scope at the end of its own
    // operation) and the next one does not exist yet.
    void dropConsumed()
    {
        if (kwBase == NULL)
            return;
        size_t page = 4096;
        size_t off = (size_t)(keyBuf - kwBase);
        size_t upto = (off / page) * page;
        // In 64 MiB steps: one madvise per key would be ~10^4 syscalls for no benefit.
        if (upto > kwDropped + (64ull << 20))
        {
            madvise(kwBase + kwDropped, upto - kwDropped, MADV_DONTNEED);
            kwDropped = upto;
        }
    }
'''


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tree = sys.argv[1]
    cu, h = os.path.join(tree, CU), os.path.join(tree, H)
    for p in (cu, h):
        if not os.path.exists(p):
            sys.exit("FAILED: no " + p)

    cu_txt, h_txt = open(cu).read(), open(h).read()
    if "s6_key_window" in cu_txt or "s6_key_window" in h_txt:
        sys.exit("FAILED: this tree already carries the lever")

    # THE WRITE PATH MUST BE THERE. Passing a filename is pointless if `close()` ignores it.
    if 'if (keyFile.compare("") != 0)' not in h_txt:
        sys.exit("FAILED: `close()` has no key-file branch, so the dealer cannot write the file "
                 "this step maps")
    # AND THE FREE IN THAT BRANCH MUST UNREGISTER: close()'s cpuFree(startPtr) relies on cpuMalloc's
    # pinned=true default; if it were false the pages stay registered and the step measures a fraction
    # of its claim. Checked in the header.
    memh = os.path.join(tree, "utils/gpu_mem.h")
    if not os.path.exists(memh) or "void cpuFree(void *h_a, bool pinned = true)" not in open(memh).read():
        sys.exit("FAILED: cpuFree's `pinned` parameter does not default to true in utils/gpu_mem.h, "
                 "so `close()`'s one-argument call would free without unregistering and the dealer's "
                 "pages would stay pinned")

    for name, old, txt in (("keygen construction", KEYGEN_OLD, cu_txt),
                           ("online construction", ONLINE_OLD, cu_txt),
                           ("key read", READ_OLD, h_txt),
                           ("cursor fields", FIELDS_OLD, h_txt)):
        if txt.count(old) != 1:
            sys.exit(f"FAILED: anchor `{name}` matched {txt.count(old)} times, expected exactly 1")

    cu_txt = cu_txt.replace(KEYGEN_OLD, KEYGEN_NEW, 1).replace(ONLINE_OLD, ONLINE_NEW, 1)
    h_txt = h_txt.replace(FIELDS_OLD, FIELDS_NEW, 1).replace(READ_OLD, READ_NEW, 1)

    if "#include <sys/mman.h>" not in h_txt:
        h_txt = h_txt.replace("#pragma once", "#pragma once\n\n#include <sys/mman.h>  // s6_key_window", 1) \
            if "#pragma once" in h_txt else "#include <sys/mman.h>  // s6_key_window\n" + h_txt

    # DROP BEFORE EVERY READ: the key-read count is asserted so a future reader added elsewhere is
    # noticed rather than silently left without a drop.
    reads = re.findall(r"\n        auto k = read", h_txt)
    if len(reads) < 5:
        sys.exit(f"FAILED: found {len(reads)} key-read sites, expected the seven this file has; "
                 "the window would be dropped at the wrong moments")
    h_txt = h_txt.replace("\n        auto k = read", "\n        dropConsumed();\n        auto k = read")

    open(cu, "w").write(cu_txt)
    open(h, "w").write(h_txt)
    print(f"patch_key_window: dealer writes, online phase maps, {len(reads)} read sites drop behind "
          f"the cursor")


if __name__ == "__main__":
    main()
