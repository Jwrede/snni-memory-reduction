#!/usr/bin/env python3
"""Names the call sites of a pmsample resident table; emits the campaign object format.

Usage: resolve_resident.py <resident.txt> <pm.maps> [objects_resident.jsonl]
                           [--exec "podman run --rm <image>"] [--top N]
                           [--program-root /root/moai ...]

Addresses resolved against the sampler's /proc/<pid>/maps snapshot.
--program-root: name a row by the first frame whose source file lies under the prefix (stable
across images, unlike a fixed SNNI_PM_DEPTH index). Needs table version C3+ and `-g`.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from symbolise import resolve  # noqa: E402


def load_maps(path):
    """(start, end, file_offset, path) for every file-backed mapping."""
    out = []
    for ln in open(path, errors="replace"):
        p = ln.split()
        if len(p) < 6 or "-" not in p[0]:
            continue
        lo, hi = p[0].split("-")
        try:
            out.append((int(lo, 16), int(hi, 16), int(p[2], 16), p[5]))
        except ValueError:
            continue
    return out


def main():
    argv = sys.argv[1:]
    if len(argv) < 2:
        sys.exit(__doc__)

    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    exec_prefix = opt("--exec", "")
    top = int(opt("--top", 15))
    program_roots = [argv[i + 1] for i, a in enumerate(argv)
                     if a == "--program-root" and i + 1 < len(argv)]
    flagged = {argv[i + 1] for i, a in enumerate(argv) if a.startswith("--") and i + 1 < len(argv)}
    pos = [a for a in argv if not a.startswith("--") and a not in flagged]

    rows, header, meta, fullscan_maps = [], "", {}, []
    for ln in open(pos[0], errors="replace"):
        if ln.startswith("#"):
            f = ln.split()
            # `# map <resident> <unaccounted> <perms> [path]` from pmsample's SNNI_PM_FULLSCAN:
            # locates memory the recorder can't name (never malloc'd, below the floor, or
            # allocator-retained) by mapping rather than as a bare subtracted total.
            if len(f) >= 5 and f[1] == "map":
                try:
                    fullscan_maps.append({"resident": int(f[2]), "unaccounted": int(f[3]),
                                          "perms": f[4], "path": " ".join(f[5:])})
                except ValueError:
                    pass
                continue
            for tok in ln[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    if k in ("sample_ts_ms", "sample_rss_kb", "drops", "live_allocations",
                             "fullscan_skipped_bytes"):
                        try:
                            meta[k] = int(v)
                        except ValueError:
                            pass
                    elif k == "mode":
                        meta["fullscan_mode"] = v
            header = header or ln.strip()
            continue
        f = ln.split()
        if len(f) >= 4:
            # 5th column: caller of the allocating function (SNNI_PM_DEPTH=2); 0/absent = not captured.
            caller = int(f[4], 16) if len(f) >= 5 else 0
            # Columns 6+: frame chain above the allocation (table version C3+), innermost first;
            # absent in older tables, which then keep their fixed-index name.
            chain = []
            for tok in f[5:]:
                try:
                    chain.append(int(tok, 16))
                except ValueError:
                    break
            rows.append((int(f[0]), int(f[1]), int(f[2]), int(f[3], 16), caller, tuple(chain)))
    if not rows:
        sys.exit("no rows in the resident table")

    maps = load_maps(pos[1])
    # Inline resolution (-i) is used only when a caller was captured: without it, an optimized
    # C++ caller's address is named after the function it was inlined FROM rather than INTO (this
    # alone collapsed 49.3% of a device peak onto one wrapper). Not enabled for depth-1 tables, to
    # avoid silently renaming rows in runs nobody re-measured.
    has_caller = any(r[4] for r in rows)
    want = {r[3] for r in rows} | {r[4] for r in rows if r[4]}
    if program_roots:
        for r in rows:
            want |= {a for a in r[5] if a}
    src = {} if program_roots else None
    names = resolve(want, maps, exec_prefix, inline=has_caller, src_out=src)

    def program_frame(r):
        """First recorded frame (walking outward) whose source file lies under a declared program
        root. Returns None for a pre-C3 table or no match, so the row keeps its fixed-index name
        rather than a confident wrong one."""
        for a in r[5]:
            if not a:
                continue
            path = (src or {}).get(a)
            if path and any(path.startswith(root) for root in program_roots):
                return a
        return None

    def label_of(r):
        """Name a row by its CALLER when captured, with the allocating function (often
        unattackable third-party code, e.g. libc10.so/python3.10) kept as context."""
        site = names.get(r[3], f"0x{r[3]:x}")
        pf = program_frame(r) if program_roots else None
        if pf:
            # Program frame leads (what a lever edits, and what stays stable across a lever
            # change); the frame just below it says what kind of allocation this is.
            below = None
            for a in r[5]:
                if a == pf:
                    break
                if a:
                    below = a
            lib = names.get(below, "") if below else ""
            return names.get(pf, f"0x{pf:x}") + ("  <- " + lib if lib else "  <- " + site)
        if not r[4]:
            return site
        return names.get(r[4], f"0x{r[4]:x}") + "  <- " + site

    tot_res = sum(r[0] for r in rows)
    print(header)
    print(f"attributed resident {tot_res/1e6:.1f} MB across {len(rows)} call sites\n")
    print(f"{'resident':>12} {'allocated':>12} {'touched':>8} {'count':>6}  site")
    objs = {}
    for r in rows[:top]:
        res, alloc, n = r[0], r[1], r[2]
        print(f"{res/1e6:11.1f}M {alloc/1e6:11.1f}M {100*res/alloc if alloc else 0:7.0f}% {n:6d}  "
              f"{label_of(r)[:88]}")
    for r in rows:
        lbl = label_of(r)
        objs[lbl] = objs.get(lbl, 0) + r[0]

    if fullscan_maps:
        un = sum(m["unaccounted"] for m in fullscan_maps)
        print(f"\nfullscan: {len(fullscan_maps)} mappings hold resident memory, "
              f"{sum(m['resident'] for m in fullscan_maps)/1e6:.1f} MB total, "
              f"{un/1e6:.1f} MB of it not covered by any recorded allocation")
        for m in sorted(fullscan_maps, key=lambda m: -m["unaccounted"])[:8]:
            if not m["unaccounted"]:
                break
            print(f"   {m['unaccounted']/1e6:9.1f} MB unnamed  {m['perms']}  "
                  f"{m['path'] or '[anonymous]'}")
        if meta.get("fullscan_skipped_bytes"):
            # Printed even though these mappings are unreadable and measured to hold nothing: a
            # bounded scan that does not say what it bounded reads exactly like a complete one.
            print(f"   ({meta['fullscan_skipped_bytes']/1e9:.1f} GB of PROT_NONE reservation "
                  f"skipped; rerun with SNNI_PM_FULLSCAN=all to scan it)")

    if len(pos) > 2:
        rec = {"rss_kb": tot_res // 1024, "source": "pagemap", "objects": objs}
        if fullscan_maps:
            rec["maps"] = fullscan_maps
            rec.update({k: meta[k] for k in ("fullscan_skipped_bytes", "fullscan_mode")
                        if k in meta})
        # Sampler is throttled, so the table can describe an instant slightly below the true peak;
        # sample_rss_kb makes that gap a published number instead of an assumption.
        rec.update({k: meta[k] for k in ("sample_ts_ms", "sample_rss_kb", "drops") if k in meta})
        with open(pos[2], "w") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"\nwrote {pos[2]}")
        if meta.get("drops"):
            print(f"WARNING: recorder dropped {meta['drops']} allocations; the table is incomplete")
        if meta.get("sample_rss_kb"):
            print(f"sampled at rss={meta['sample_rss_kb']} kB "
                  f"({100*tot_res/1024/meta['sample_rss_kb']:.0f}% attributed)")


if __name__ == "__main__":
    main()
