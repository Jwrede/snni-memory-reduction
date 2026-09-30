#!/usr/bin/env python3
"""s6_dealer_window: the dealer writes out and rewinds after every operation; buffer holds one
operation's key (not 16.83 GiB). All keyBuf sites append, nothing reads back. close() assert bounds
the window (1 GiB, 1.30x the largest operation, 787 MiB, p_keywindow_high). Key bytes, order and
file unchanged. Paid (same 17 GB written).

    patch_dealer_window.py <sigma source tree>
"""

import os
import re
import sys

CU = "experiments/sigma/sigma.cu"
H = "backend/sigma.h"

# ---- the buffer shrinks from the whole key stream to one operation ----
# Two-line anchor (assignment + bert-base's net construction) so it matches only the bert-base
# branch among the eight keyBufSz lines.
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

    # THE PROBE MUST HAVE RUN FIRST: this buffer size and its eight call sites are p_keywindow_high's;
    # without the probe kwNote() is uncalled and the dealer fills 1 GiB and aborts.
    if "snniKwHigh" not in h:
        sys.exit("FAILED: the key-window probe is not in this tree, so there are no call sites to "
                 "rewind at and the 1 GiB buffer would overflow on the first large operation")
    if h.count("kwNote();") != 8:
        sys.exit(f"FAILED: {h.count('kwNote();')} rewind sites found, expected the probe's 8")
    # AND THE ONLINE PHASE MUST ALREADY MAP THE FILE, or the dealer writes something nobody reads.
    if "mmap(NULL, keySize" not in h:
        sys.exit("FAILED: the online phase does not map the key file (patch_key_window missing), so "
                 "this step would write a file and then read it into a 17 GiB buffer anyway")
    # <fstream> is reachable transitively via close()'s std::ofstream, but added explicitly so an
    # upstream refactor cannot turn a transitive include into a build error read as this patch's fault.
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
