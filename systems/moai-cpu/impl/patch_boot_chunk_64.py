#!/usr/bin/env python3
"""MOAI-CPU boot_chunk_64: bootstrap 64 ciphertexts per chunk instead of 128.

    patch_boot_chunk_64.py <moai source tree>

After the free line and park_boot_layer the per-chunk bootstrap working set (BSGS, batch-split and
polynomial-evaluation temporaries, ~18 GB at 128 in flight) is the largest reducible object at the
bootstrap peaks (phase keys 19.8 GB = target key stage). Halving the chunk halves the temporaries and
doubles phase-key reloads (12 instead of 6): paid. Bit-identical (same code, same order).
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"
OLD = "#define BOOT_CHUNK_SIZE 128\n"
NEW = "#define BOOT_CHUNK_SIZE 64\n"
MARK_OLD = 'fprintf(stderr,"LEVER|chunked_bootstrap|size=%d\\n", BOOT_CHUNK_SIZE);'
MARK_NEW = ('fprintf(stderr,"LEVER|chunked_bootstrap|size=%d\\n", BOOT_CHUNK_SIZE); '
            'fprintf(stderr,"LEVER|boot_chunk_64|size=%d\\n", BOOT_CHUNK_SIZE);')


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "boot_chunk_64" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(OLD) != 1:
        sys.exit(f"FAILED: the chunk-size define appears {txt.count(OLD)} times, expected 1")
    if txt.count(MARK_OLD) != 1:
        sys.exit(f"FAILED: the chunked_bootstrap marker appears {txt.count(MARK_OLD)} times, expected 1")
    txt = txt.replace(OLD, NEW, 1).replace(MARK_OLD, MARK_NEW, 1)
    open(src, "w").write(txt)
    if "#define BOOT_CHUNK_SIZE 64" not in open(src).read():
        sys.exit("FAILED: applied but define absent")
    print("patch_boot_chunk_64: BOOT_CHUNK_SIZE 128 -> 64 (" + SRC + ")")


if __name__ == "__main__":
    main()
