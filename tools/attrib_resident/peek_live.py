#!/usr/bin/env python3
"""Reads a pmrec table during the run and prints the recorded frames (read-only).

    peek_live.py <results dir with pmtable_* and the pid file> [pid-file-name]

Purpose: check SNNI_PM_DEPTH early in a 13 to 23 h run.
Run on the node of the process:
    srun --jobid=<job> --overlap -n1 python3 peek_live.py <results dir>
Static binary: `site`/`caller` may show only the executable; resolve with
    addr2line -e /proc/<pid>/exe -f -C -i $((addr - load_base))
"""
import collections
import glob
import os
import struct
import sys

# Two layouts, for the same reason pmsample.c reads both: a run started under the older recorder
# must still be peekable while it's going. C3 carries the frame chain, C2 does not.
MAGIC_C3 = 0x504D52454333
MAGIC_C2 = 0x504D52454332
PM_FRAMES = 12
SLOT_C3 = struct.Struct("<QQQQ" + "Q" * PM_FRAMES)  # ptr, size, site, caller, frames[12]
SLOT_C2 = struct.Struct("<QQQQ")                    # ptr, size, site, caller
HDR = struct.Struct("<QQQQ")   # magic, inserts, drops, slots


def load_ranges(pid):
    path = "/proc/%s/maps" % pid
    if not os.path.exists(path):
        sys.exit("FAILED: no %s. This has to run on the node that owns the process, e.g. "
                 "`srun --jobid=<job> --overlap -n1 python3 %s ...`" % (path, sys.argv[0]))
    out = []
    for ln in open(path):
        p = ln.split()
        if len(p) < 6 or "x" not in p[1]:
            continue
        lo, hi = (int(x, 16) for x in p[0].split("-"))
        out.append((lo, hi, os.path.basename(p[5])))
    return out


def main():
    if not 2 <= len(sys.argv) <= 3:
        sys.exit(__doc__)
    d = sys.argv[1]
    pidfile = sys.argv[2] if len(sys.argv) == 3 else None
    if pidfile is None:
        cand = [f for f in os.listdir(d) if f.endswith(".pid")]
        if len(cand) != 1:
            sys.exit("FAILED: expected exactly one *.pid in %s, found %d; name it as the second "
                     "argument" % (d, len(cand)))
        pidfile = cand[0]
    pid = open(os.path.join(d, pidfile)).read().strip()
    rng = load_ranges(pid)

    def who(a):
        for lo, hi, n in rng:
            if lo <= a < hi:
                return n
        return "?"

    agg = collections.Counter()
    total = 0
    tables = 0
    for f in sorted(glob.glob(os.path.join(d, "pmtable_*"))):
        b = open(f, "rb").read()
        if len(b) < HDR.size:
            continue
        magic, _ins, _drops, slots = HDR.unpack_from(b, 0)
        if magic == MAGIC_C3:
            SLOT = SLOT_C3
        elif magic == MAGIC_C2:
            SLOT = SLOT_C2
        else:
            continue  # not ready yet, or a table this reader must not interpret
        tables += 1
        n = min(slots, (len(b) - HDR.size) // SLOT.size)
        for i in range(n):
            fields = SLOT.unpack_from(b, HDR.size + i * SLOT.size)
            ptr, size, site, caller = fields[:4]
            if ptr:
                agg[(who(site), who(caller), caller)] += size
                total += size

    if not tables:
        sys.exit("FAILED: no readable pmtable in %s. Either the recorder has not published one "
                 "yet, or its magic is not the layout this reader knows." % d)
    print("tables=%d  live bytes=%.1f MB   (a LIVE reading, not the peak)" % (tables, total / 1e6))
    for (s, c, addr) in sorted(agg, key=lambda k: -agg[k])[:12]:
        print("  %10.1f MB   site=%-20s caller=%-20s 0x%x" % (agg[(s, c, addr)] / 1e6, s, c, addr))


if __name__ == "__main__":
    main()
