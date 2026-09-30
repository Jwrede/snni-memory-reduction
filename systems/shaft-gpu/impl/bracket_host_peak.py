#!/usr/bin/env python3
"""Locates the transient host peak between two phase markers from monotone VmHWM (the interval of
the last rise contains the peak). Locator only. Input: the run's MEM| marker stream.

Usage:  bracket_host_peak.py <run_dir_or_markers_file> [--party N]
"""
import os
import sys


def read_markers(path, party):
    """(index, layer, phase, op, rss, ts, hwm) for one party, in file order."""
    out = []
    with open(path, errors="replace") as f:
        for i, line in enumerate(f):
            if not line.startswith("MEM|"):
                continue
            p = line.rstrip("\n").split("|")
            if len(p) < 8:
                continue  # a pre-2026-08-01 marker, which carries no VmHWM
            try:
                rank, rss, ts, hwm = int(p[1]), int(p[5]), int(p[6]), int(p[7])
            except ValueError:
                continue
            if party is not None and rank != party:
                continue
            out.append((i, p[2], p[3], p[4], rss, ts, hwm))
    return out


def bracket(marks):
    """Every interval in which VmHWM rose, largest last."""
    rises = []
    prev = None
    for m in marks:
        if prev is not None and m[6] > prev[6]:
            rises.append((prev, m, m[6] - prev[6]))
        prev = m
    return rises


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target = sys.argv[1]
    party = None
    if "--party" in sys.argv:
        party = int(sys.argv[sys.argv.index("--party") + 1])

    path = target
    if os.path.isdir(target):
        for cand in ("n12_seq128/all_markers.txt", "all_markers.txt",
                     "n12_seq128/party0_markers.txt"):
            if os.path.exists(os.path.join(target, cand)):
                path = os.path.join(target, cand)
                break
        else:
            sys.exit(f"no marker file under {target}")

    marks = read_markers(path, party)
    if not marks:
        sys.exit(f"no VmHWM-carrying MEM markers in {path}. Runs before 2026-08-01 do not have "
                 f"the field, and a run without it cannot be bracketed, only polled.")

    peak = marks[-1][6]
    rises = bracket(marks)
    print(f"markers: {len(marks)}   party: {party if party is not None else 'all'}")
    print(f"VmHWM at the last marker: {peak} kB")
    if not rises:
        print("VmHWM never rose across the marker stream: the peak predates the first marker.")
        return
    print()
    print("intervals in which VmHWM rose, by size:")
    print(f"{'delta kB':>12}  {'share':>6}  {'ms':>6}  between")
    for a, b, d in sorted(rises, key=lambda r: -r[2])[:15]:
        share = d / peak if peak else 0.0
        print(f"{d:>12}  {share:>6.1%}  {b[5]-a[5]:>6}  "
              f"{a[3]}[{a[1]}] -> {b[3]}[{b[1]}]")
    last = rises[-1]
    print()
    print("THE PEAK: the last rise, which is the interval that set the high-water mark")
    print(f"  from {last[0][3]} (layer {last[0][1]}, phase {last[0][2]})")
    print(f"  to   {last[1][3]} (layer {last[1][1]}, phase {last[1][2]})")
    print(f"  +{last[2]} kB in {last[1][5]-last[0][5]} ms, "
          f"RSS {last[0][4]} -> {last[1][4]} kB, VmHWM now {last[1][6]} kB")
    poll_gap = last[1][6] - max(m[4] for m in marks)
    print(f"  highest RSS any marker OBSERVED: {max(m[4] for m in marks)} kB, "
          f"{poll_gap} kB ({poll_gap/last[1][6]:.1%}) below the high-water mark")


if __name__ == "__main__":
    main()
