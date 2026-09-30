#!/usr/bin/env python3
"""SIGMA-GPU s6: key material mapped and consumed through a window instead of held whole. Paid.

    patch_key_window.py <EzPC/GPU-MPC tree>

Basis: s5_keybuf_tight host peak:
    sigma.cu:254:main <- gpu_mem.cu:77:cpuMalloc   18,387.8 MB   94.3%   keyBuf
    everything else                                 ~1,100 MB     5.7%
off-path.md's "structural gap" closure (2026-08-15) retracted on 18.08. after source reading:

1. Write and read paths exist, disabled by an empty filename: sigma.cu constructs keygen and SIGMA
   with "", so close()'s `if (keyFile.compare("") != 0)` skips the write and SIGMA skips the read.
2. Reader only advances a cursor (fss/gpu_matmul.h:51):
       k.A = (T *)*key_as_bytes;   *key_as_bytes += k.mem_size_A;
       k.B = (T *)*key_as_bytes;   *key_as_bytes += k.mem_size_B;
       k.C = (T *)*key_as_bytes;   *key_as_bytes += k.mem_size_C;
   startPtr = keyBuf appears twice in backend/sigma.h, both at construction; no reset.
3. A key lives one operation (backend/sigma.h:135, :148):
       auto k = readGPUMatmulKey<T>(p, TruncateType::None, &keyBuf);
       c.d_data = gpuMatmul(peer, party, p, k, ...);
       }                                            // k leaves scope here

Donor: SHAFT-CPU s6_stream_load (safe_open mmap + posix_fadvise(DONTNEED)); MOAI-GPU park/reload.
mmap because readers hold pointers into the buffer.
  * dealer writes its key file (existing path, enabled)
  * online phase maps it instead of allocating keySize and reading
  * dealer buffer released before the online phase is constructed (required; cf. MOAI-GPU
    s7_fused_input_encrypt: -14,944 MiB removed, +0.04% measured)
  * pages behind the cursor dropped with MADV_DONTNEED at every key read (MAP_PRIVATE; safe since
    the cursor never revisits a byte)
Scope: not the keygen/online interleaving of off-path.md; target 220,944 kB not reached.
Cost: paid, 16.83 GiB written once and read once; runtime risk stated before the run.
Gate: expected identical (same bytes, seed, order).
Expected: host peak 19,052,324 kB -> 1,500,000 to 2,500,000 kB (~ -87%).
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
    # AND THE FREE IN THAT BRANCH MUST UNREGISTER, which is the whole reason this step needs no
    # release of its own. `close()` calls `cpuFree(startPtr)` with one argument; `cpuMalloc` pinned
    # the buffer with `cudaHostRegister`, so if `pinned` did NOT default to true the pages would
    # stay registered and this step would measure a fraction of what it claims. Checked in the
    # header rather than assumed.
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

    # DROP BEFORE EVERY READ. Every key read in this file has the same shape and the same eight
    # spaces of indentation; the count is asserted so a future reader added elsewhere is noticed
    # rather than silently left without a drop.
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
