#!/usr/bin/env python3
"""Python census -> campaign object record.

    pycensus_objects.py <pycensus_pid*.txt> <out.jsonl> [--peak-kb N]

Uses the census block at the highest RSS; output composed by smaps_objects.py --heap. Coverage
~95% of the recorder's RSS (both carried). Aliased objects (torch.from_numpy views) dropped.
"""
import collections
import json
import sys


def main():
    argv = sys.argv[1:]
    if len(argv) < 2:
        sys.exit(__doc__)
    src, out = argv[0], argv[1]
    peak_kb = int(argv[argv.index("--peak-kb") + 1]) if "--peak-kb" in argv else 0

    best = None
    cur = None
    for ln in open(src, errors="replace"):
        if ln.startswith("PYCENSUS|"):
            try:
                rss = int(ln.split("rss_kb=")[1].split("|")[0])
            except (IndexError, ValueError):
                continue
            cur = {"rss": rss, "rows": [], "hdr": ln.strip()}
            if best is None or rss > best["rss"]:
                best = cur
        elif cur is not None and ln.startswith("PYOBJ"):
            f = ln.split()
            if len(f) < 7:
                continue
            try:
                res = int(f[1])
            except ValueError:
                continue
            alias = f[-2]
            held = f[-3].replace("held_by=", "")
            kind = f[3]
            detail = " ".join(f[4:-3])
            cur["rows"].append((res, kind, detail, held, alias))

    if best is None:
        sys.exit(f"no census block in {src}")

    # Grouped by (kind, dtype, holder), not shape: shape fails the campaign's group test (one
    # change must lift the whole group) when one object spans several shapes. Shapes are kept
    # per group as evidence a reader can sanity-check (e.g. recognising a BERT by its shapes).
    objs = collections.defaultdict(int)
    shapes = collections.defaultdict(collections.Counter)
    for res, kind, detail, held, alias in best["rows"]:
        if alias == "alias":
            continue
        dtype = detail.split("(")[0] if "(" in detail else detail
        shape = detail[len(dtype):] if "(" in detail else ""
        # No `heap:` prefix: smaps_objects.py adds it (adding it twice broke a prior record).
        key = f"{kind}:{dtype} held_by={held}"
        objs[key] += res
        if shape:
            shapes[key][shape] += 1

    rec = {
        "rss_kb": best["rss"],
        "source": "pycensus",
        "census_rss_kb": best["rss"],
        "objects": dict(objs),
        "shapes": {k: dict(v.most_common(8)) for k, v in shapes.items()},
    }
    if peak_kb:
        rec["census_vs_peak_pct"] = round(100.0 * best["rss"] / peak_kb, 1)
    with open(out, "w") as fh:
        fh.write(json.dumps(rec) + "\n")
    tot = sum(objs.values())
    print(f"{src}: block at rss={best['rss']} kB, {len(objs)} groups, {tot/1e6:.1f} MB")
    if peak_kb:
        print(f"   census sat at {100.0 * best['rss'] / peak_kb:.1f}% of the polled peak")
    for k, v in sorted(objs.items(), key=lambda kv: -kv[1])[:8]:
        top = ", ".join(f"{n}x{sh}" for sh, n in shapes[k].most_common(3))
        print(f"   {v/1e6:9.1f} MB  {k[:60]:60s} {top}")


if __name__ == "__main__":
    main()
