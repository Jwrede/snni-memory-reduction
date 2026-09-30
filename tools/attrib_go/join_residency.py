#!/usr/bin/env python3
"""Joins a parsed Go heap dump with a pagemap snapshot: resident bytes per allocation site.

    join_residency.py <objects.jsonl> <pagemap.bin> <out.jsonl> [sample_rss_kb] [--program-root PREFIX]

--program-root: key by the first stack frame starting with PREFIX (ARION: `Arion/`); else the
allocation site.
Output: one line in resolve_resident.py's format, consumed by smaps_objects.py --heap:
    {"rss_kb": <sample_rss_kb>, "source": "goheapdump+pagemap", "sample_rss_kb": ...,
     "objects": {"<site>": resident_bytes, ...}}
Accounting per page overlap, byte-accurate. Unsampled objects -> "unsampled". Present total minus
joined total -> "go:runtime_retained" (spans, freed-but-resident, stacks).
"""
import bisect
import json
import struct
import sys

PAGE = 4096


def load_snapshot(path):
    starts, bitmaps, npages = [], [], []
    with open(path, "rb") as fh:
        while True:
            hdr = fh.read(16)
            if len(hdr) < 16:
                break
            lo, n = struct.unpack("<QQ", hdr)
            bits = fh.read((n + 7) // 8)
            starts.append(lo)
            bitmaps.append(bits)
            npages.append(n)
    return starts, bitmaps, npages


def main():
    argv = sys.argv[1:]
    root = None
    if "--program-root" in argv:
        i = argv.index("--program-root")
        root = argv[i + 1]
        del argv[i:i + 2]
    objs_path, snap_path, out = argv[0], argv[1], argv[2]
    rss_kb = int(argv[3]) if len(argv) > 3 else 0

    def key(r):
        if root:
            for f in r.get("stack", []):
                if f.startswith(root):
                    return f
        return r["site"]
    starts, bitmaps, npages = load_snapshot(snap_path)
    present_total = sum(
        bin(int.from_bytes(b, "little")).count("1") for b in bitmaps) * PAGE

    def resident_bytes(addr, size):
        got = 0
        end = addr + size
        i = bisect.bisect_right(starts, addr) - 1
        while i >= 0 and i < len(starts) and starts[i] < end:
            lo, bits, n = starts[i], bitmaps[i], npages[i]
            hi = lo + n * PAGE
            if hi <= addr:
                i += 1
                if i >= len(starts) or starts[i] >= end:
                    break
                continue
            a, b = max(addr, lo), min(end, hi)
            p0, p1 = (a - lo) // PAGE, (b - lo + PAGE - 1) // PAGE
            for p in range(p0, p1):
                if bits[p // 8] >> (p % 8) & 1:
                    plo, phi = lo + p * PAGE, lo + (p + 1) * PAGE
                    got += min(end, phi) - max(addr, plo)
            i += 1
        return got

    agg = {}
    joined = 0
    n_obj = 0
    for ln in open(objs_path):
        r = json.loads(ln)
        n_obj += 1
        res = resident_bytes(r["addr"], r["size"])
        if not res:
            continue
        joined += res
        k = key(r)
        agg[k] = agg.get(k, 0) + res
    retained = max(0, present_total - joined)
    if retained:
        agg["go:runtime_retained"] = retained
    rec = {"rss_kb": rss_kb, "source": "goheapdump+pagemap",
           "sample_rss_kb": rss_kb, "objects": agg}
    with open(out, "w") as o:
        o.write(json.dumps(rec) + "\n")
    print(f"JOIN|objects={n_obj}|joined_bytes={joined}|anon_present={present_total}"
          f"|runtime_retained={retained}")
    top = sorted(((v, k) for k, v in agg.items()), reverse=True)[:8]
    for v, k in top:
        print(f"  {v/1e6:9.2f} MB  {k}")


if __name__ == "__main__":
    main()
