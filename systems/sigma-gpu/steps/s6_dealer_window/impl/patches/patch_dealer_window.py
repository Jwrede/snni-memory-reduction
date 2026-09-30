#!/usr/bin/env python3
"""s6_dealer_window: the dealer writes out and rewinds after every operation; buffer holds one.

Origin: off-path-runs/s6_key_window (online phase maps the key file): object 18,387.8 MB -> 134.2 MB
(-99.3%), online phase 1.09 GB instead of 18.9 GB for 60 s; published peak -0.71% (VmHWM set by the
dealer's earlier fill; build refused: "host objects sampled at 2,060,672 kB, 89.1% below the
18,916,820 kB peak"). This step targets the dealer's buffer.
Rewind validity: all eight keyBuf sites append (seven pass &keyBuf to an advancing generator, one
`keyBuf += memSz` after moveIntoCPUMem); startPtr read only in keySize = keyBuf - startPtr and the
final file write; nothing reads back. Bytes reach the file in the same order.
Window size (p_keywindow_high, no allocation change):
    SNNI_KW|op_high=825125000|ops=97|keysize=18075946656
787 MiB vs 16.83 GiB (21.9x), 97 operations, mean 178 MiB. Buffer 1 GiB (1.30x the largest
operation); close() assert (live, s5_keybuf_tight) rewritten to bound the window.
Unchanged: key material, order, size, the mapped file. Gate COMM:<bytes> KEYS:<bytes>.
Paid: same 17 GB written as s6_key_window; write spread over the dealer's compute.
"""

import os
import re
import sys

CU = "experiments/sigma/sigma.cu"
H = "backend/sigma.h"

# ---- the buffer shrinks from the whole key stream to one operation ----
# The assignment sits inside the model's own branch and is indented eight spaces, not four, and
# there are eight `keyBufSz = ... * OneGB;` lines in this file for eight model sizes. The anchor
# therefore carries the value AND the line that follows it, which is bert-base's network
# construction, so it can only match this branch. (The first version of this patch anchored on
# `    size_t keyBufSz = 17 * OneGB;` and matched zero times; the build refused at exit 14, which is
# the guard doing its job, but it cost a build slot.)
BUF_OLD = """        keyBufSz = 17 * OneGB;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);"""
BUF_NEW = """        // s6_dealer_window: the dealer rewinds after every operation, so this holds ONE
        // operation's key rather than all 97. `p_keywindow_high` measured the largest single
        // operation at 825,125,000 B = 787 MiB; 1 GiB is a margin of 1.30x, and that margin is
        // CHECKED by the assert at the end of close() rather than trusted.
        keyBufSz = 1 * OneGB;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);"""

# ---- the stream, opened once, and the counter of what has already left ----
FIELDS_OLD = '''    size_t snniKwHigh = 0, snniKwOps = 0;
    u8 *kwMark = NULL;
    void kwNote()
    {
        size_t d = (size_t)(keyBuf - kwMark);
        if (d > snniKwHigh) snniKwHigh = d;
        kwMark = keyBuf;
        snniKwOps++;
    }
'''
FIELDS_NEW = '''    size_t snniKwHigh = 0, snniKwOps = 0;
    u8 *kwMark = NULL;
    // s6_dealer_window: what has already been written out, and the sink it went to.
    size_t kwWritten = 0;
    std::ofstream kwOut;
    // Called after every operation that appends to keyBuf. It writes the filled prefix and resets
    // the cursor, so the buffer never holds more than one operation's key.
    //
    // It keeps the probe's high-water accounting, because that number is what sized the buffer and
    // a run whose largest operation exceeded the measured one should be visible rather than
    // inferred from an abort.
    void kwNote()
    {
        size_t d = (size_t)(keyBuf - kwMark);
        if (d > snniKwHigh) snniKwHigh = d;
        snniKwOps++;
        if (!kwOut.is_open())
        {
            // No key file: the dealer must keep everything, exactly as the stock program does.
            kwMark = keyBuf;
            return;
        }
        size_t n = (size_t)(keyBuf - startPtr);
        if (n)
        {
            kwOut.write((char *)startPtr, n);
            kwWritten += n;
            keyBuf = startPtr;
        }
        kwMark = keyBuf;
    }
'''

INIT_OLD = '''        keyBuf = cpuMalloc(keyBufSize);
        startPtr = keyBuf;
        kwMark = keyBuf;
'''
INIT_NEW = '''        keyBuf = cpuMalloc(keyBufSize);
        startPtr = keyBuf;
        kwMark = keyBuf;
        // s6_dealer_window: the sink the rewinds write into. Opened here rather than at close(),
        // because close() is now only the tail.
        if (keyFile.compare("") != 0)
            kwOut.open(keyFile + "_" + std::to_string(party) + ".dat",
                       std::ios::binary | std::ios::trunc);
'''

# ---- close() writes the tail instead of the whole thing ----
CLOSE_OLD = '''        /*size_t*/ keySize = keyBuf - startPtr;
        size_t padding = 4096 - (keySize % 4096);
        char *zeros = new char[padding];
        memset(zeros, 0, padding);
        memcpy(keyBuf, zeros, padding);
        keyBuf += padding;
        keySize += padding;
        assert(keySize < keyBufSize);
        if (keyFile.compare("") != 0)
        {
            std::ofstream f(keyFile + "_" + std::to_string(party) + ".dat");
            f.write((char *)startPtr, keySize);
            f.close();
            cpuFree(startPtr);
        }
'''
CLOSE_NEW = '''        // s6_dealer_window: keySize is the RUNNING TOTAL now, because most of it is already on
        // disk. Only the tail since the last rewind is still in the buffer.
        size_t tail = (size_t)(keyBuf - startPtr);
        /*size_t*/ keySize = kwWritten + tail;
        size_t padding = 4096 - (keySize % 4096);
        char *zeros = new char[padding];
        memset(zeros, 0, padding);
        memcpy(keyBuf, zeros, padding);
        keyBuf += padding;
        keySize += padding;
        // THE BOUND IS ON THE WINDOW, not on the total. The total no longer exists in memory at any
        // instant, so the old assert would compare a 17 GB number against a 1 GB buffer and abort on
        // a correct run. What must hold is that one operation plus its padding fits.
        assert((size_t)(keyBuf - startPtr) < keyBufSize);
        if (kwOut.is_open())
        {
            kwOut.write((char *)startPtr, tail + padding);
            kwOut.close();
            cpuFree(startPtr);
        }
'''


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: patch_dealer_window.py <source tree>")
    tree = sys.argv[1]
    cup, hp = os.path.join(tree, CU), os.path.join(tree, H)
    for f in (cup, hp):
        if not os.path.exists(f):
            sys.exit(f"FAILED: {f} does not exist")
    cu, h = open(cup).read(), open(hp).read()

    if "kwWritten" in h:
        sys.exit("FAILED: this tree already carries the lever")

    # THE PROBE MUST HAVE RUN FIRST. This step's buffer size is `p_keywindow_high`'s number, and
    # its eight call sites are the probe's. Applying it to a tree without the probe would leave
    # `kwNote()` uncalled and the dealer would fill 1 GiB and abort.
    if "snniKwHigh" not in h:
        sys.exit("FAILED: the key-window probe is not in this tree, so there are no call sites to "
                 "rewind at and the 1 GiB buffer would overflow on the first large operation")
    if h.count("kwNote();") != 8:
        sys.exit(f"FAILED: {h.count('kwNote();')} rewind sites found, expected the probe's 8")
    # AND THE ONLINE PHASE MUST ALREADY MAP THE FILE, or the dealer writes something nobody reads.
    if "mmap(NULL, keySize" not in h:
        sys.exit("FAILED: the online phase does not map the key file (patch_key_window missing), so "
                 "this step would write a file and then read it into a 17 GiB buffer anyway")
    # `std::ofstream` is already used by close() in this file, so <fstream> is reachable here
    # (transitively, like the `std::filesystem` the online phase uses). It is added explicitly
    # anyway: relying on a transitive include is how a refactor upstream turns into a build error
    # that reads as if this patch were wrong.
    if "#include <fstream>" not in h:
        h = h.replace("#include <sys/mman.h>  // s6_key_window",
                      "#include <sys/mman.h>  // s6_key_window\n#include <fstream>    // s6_dealer_window", 1)

    for name, old, txt in (("key buffer size", BUF_OLD, cu),
                           ("probe counter", FIELDS_OLD, h),
                           ("buffer init", INIT_OLD, h),
                           ("close body", CLOSE_OLD, h)):
        if txt.count(old) != 1:
            sys.exit(f"FAILED: anchor `{name}` matched {txt.count(old)} times, expected exactly 1")

    cu = cu.replace(BUF_OLD, BUF_NEW, 1)
    h = h.replace(FIELDS_OLD, FIELDS_NEW, 1).replace(INIT_OLD, INIT_NEW, 1) \
         .replace(CLOSE_OLD, CLOSE_NEW, 1)

    open(cup, "w").write(cu)
    open(hp, "w").write(h)
    print("patch_dealer_window: 8 rewind sites active, buffer 17 GiB -> 1 GiB, close() writes the "
          "tail, the assert bounds the window")


if __name__ == "__main__":
    main()
