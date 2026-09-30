#!/usr/bin/env python3
"""MOAI-CPU pool_threshold_256k: free-on-return threshold 4 MiB -> 256 KiB.

    patch_pool_threshold_256k.py <moai source tree>

After m1 the pool keeps only blocks < 4 MiB (sealpool:retained 8.2 GB at the bootstrap peaks after the
input spill: plaintexts, per-limb temporaries, poly-evaluation scratch). Same bytes and operations
(caching decision only); gate unchanged. Paid (more kernel traffic; knob). Ranked by rule 6 against
chunk 64 and a finer key phase split.
"""
import os
import sys

MEMPOOL_H = os.path.join("thirdparty", "SEAL-4.1-bs", "native", "src", "seal", "util", "mempool.h")
OLD = "#define SNNI_POOL_THRESHOLD_DEFAULT (std::size_t(4) * 1024 * 1024)\n"
NEW = "#define SNNI_POOL_THRESHOLD_DEFAULT (std::size_t(256) * 1024)  /* pool_threshold_256k */\n"
MEMPOOL_C = os.path.join("thirdparty", "SEAL-4.1-bs", "native", "src", "seal", "util", "mempool.cpp")
MARK_OLD = 'std::fprintf(stderr, "LEVER|pool_threshold|bytes=%llu|source=%s\\n",'
MARK_NEW = ('std::fprintf(stderr, "LEVER|pool_threshold_256k|default=%llu\\n", (unsigned long long)SNNI_POOL_THRESHOLD_DEFAULT); '
            'std::fprintf(stderr, "LEVER|pool_threshold|bytes=%llu|source=%s\\n",')


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    h = os.path.join(sys.argv[1], MEMPOOL_H)
    c = os.path.join(sys.argv[1], MEMPOOL_C)
    for p in (h, c):
        if not os.path.exists(p):
            sys.exit("FAILED: no " + p)
    th = open(h).read()
    tc = open(c).read()
    if "pool_threshold_256k" in th or "pool_threshold_256k" in tc:
        sys.exit("FAILED: this tree already carries the lever")
    if th.count(OLD) != 1:
        sys.exit(f"FAILED: the threshold define appears {th.count(OLD)} times, expected 1")
    if tc.count(MARK_OLD) != 1:
        sys.exit(f"FAILED: the pool_threshold marker appears {tc.count(MARK_OLD)} times, expected 1")
    open(h, "w").write(th.replace(OLD, NEW, 1))
    open(c, "w").write(tc.replace(MARK_OLD, MARK_NEW, 1))
    print("patch_pool_threshold_256k: SNNI_POOL_THRESHOLD_DEFAULT 4 MiB -> 256 KiB (" + MEMPOOL_H + ")")


if __name__ == "__main__":
    main()
