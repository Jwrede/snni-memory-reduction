#!/usr/bin/env python3
"""Keep every pool-table sample (hard-linked to peaks/pool_<ts>.txt) instead of only the last, so a row can be followed across phases to tell a live object from a stale attribution. FREE, changes no measurement.
"""
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
P = sys.argv[1]
if not os.path.isfile(P):
    sys.exit(f"FAILED: {P} is not a file")

s = open(P, encoding="utf-8").read()

MARK = "pool_$ts.txt"
if MARK in s:
    print("  already present: pool trace")
    sys.exit(0)

OLD = '''                mv "$OUTDIR/peaks/resident_pool.txt.tmp" "$OUTDIR/peaks/resident_pool.txt"
'''
NEW = '''                mv "$OUTDIR/peaks/resident_pool.txt.tmp" "$OUTDIR/peaks/resident_pool.txt"
                # KEEP THIS SAMPLE, not only the last one. `link` writes a directory entry and
                # nothing else, so the frozen window is not measurably longer; the next sample
                # arrives as a fresh inode via `mv`, so this link keeps its own bytes. The peak
                # table stays exactly where it was and is still what gets published.
                ln -f "$OUTDIR/peaks/resident_pool.txt" \\
                      "$OUTDIR/peaks/pool_$ts.txt" 2>/dev/null || true
'''
if s.count(OLD) != 1:
    sys.exit(f"FAILED: the pool sample mv matched {s.count(OLD)} times in {P}, expected exactly 1")
s = s.replace(OLD, NEW, 1)

if 'sed -i "1i # sample_ts_ms=$ts' not in s:
    sys.exit("FAILED: the sample header line is not where the trace's naming argument needs it")

open(P, "w", encoding="utf-8").write(s)
print(f"  patched: every pool sample kept as peaks/pool_<ts>.txt ({P})")
