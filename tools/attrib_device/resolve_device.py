#!/usr/bin/env python3
"""Names the call sites of a cudarec device table; emits the campaign object format.

Usage: resolve_device.py <device_table.txt> <device.maps> [objects_vram.jsonl]
                         [--exec "apptainer exec <image>"] [--top N]

Symboliser shared with attrib_resident/resolve_resident.py; addresses resolved against the paired maps.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from symbolise import resolve  # noqa: E402


def s64(tok):
    """Read one recorder field as SIGNED 64-bit: a site with more frees than allocations wraps a
    uint64_t counter to ~2^64, which read unsigned would swamp every aggregate. Reading signed
    doesn't fix the recorder (see caller) but makes the defect visible instead of catastrophic."""
    v = int(tok)
    return v - (1 << 64) if v >= (1 << 63) else v


def load_maps(path):
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
    # --inline names each site by the function the allocation was inlined into, rather than the
    # CUDA error-check wrapper that otherwise tops the ranking and owns nothing.
    inline = "--inline" in argv
    flagged = {argv[i + 1] for i, a in enumerate(argv)
           if a.startswith("--") and a != "--inline" and i + 1 < len(argv)}
    pos = [a for a in argv if not a.startswith("--") and a not in flagged]

    rows, meta, hooks = [], {}, {}
    for ln in open(pos[0], errors="replace"):
        if ln.startswith("#"):
            f = ln[1:].split()
            if f and f[0] == "hooks":
                for tok in f[1:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        hooks[k] = int(v)
                continue
            for tok in f:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k] = v
            continue
        p = ln.split()
        if len(p) >= 4:
            # 5th column: caller of the allocating function (SNNI_CUDAREC_DEPTH=2); 0/absent = not captured.
            caller = int(p[4], 16) if len(p) >= 5 else 0
            # 6th column: one frame further out (SNNI_CUDAREC_DEPTH=3); 0/absent = not captured.
            caller2 = int(p[5], 16) if len(p) >= 6 else 0
            rows.append((s64(p[0]), s64(p[1]), s64(p[2]), int(p[3], 16), caller, caller2))

    # An empty table (all hooks dead) is this campaign's standing failure shape -- not an error,
    # an absence that reads like a result -- so refuse rather than emit it silently.
    fired = {k: v for k, v in hooks.items() if v}
    if hooks:
        print("hooks that fired: " + (", ".join(f"{k}={v}" for k, v in fired.items()) or "NONE"))
        dead = [k for k, v in hooks.items() if not v]
        if dead:
            print("hooks that never fired: " + ", ".join(dead))
    if not rows:
        sys.exit("device table has no rows. If no hook fired, the interposition set is wrong for "
                 "this binary: check `nm -D -u <binary> | grep -i cudaMalloc` -- a build using "
                 "--default-stream per-thread imports the _ptsz spellings instead.")

    # Sites with a negative balance (recorder defect, more frees than allocations) are excluded
    # from ranking and totals but printed so the defect stays visible.
    neg = [r for r in rows if r[0] < 0 or r[2] < 0]
    rows = [r for r in rows if r[0] >= 0 and r[2] >= 0]
    if not rows:
        sys.exit("every site in the device table has a negative balance; the recorder is broken "
                 "for this run and nothing here can be published.")

    maps = load_maps(pos[1])
    wanted = {r[3] for r in rows} | {r[3] for r in neg}
    wanted |= {r[4] for r in rows if r[4]} | {r[4] for r in neg if r[4]}
    wanted |= {r[5] for r in rows if len(r) > 5 and r[5]} | {r[5] for r in neg if len(r) > 5 and r[5]}
    names = resolve(wanted, maps, exec_prefix, inline=inline)

    def label_of(r):
        """Name a row by its CALLER when captured (the real owner), with the allocating helper
        kept as context -- the helper alone is often unattackable (e.g. make_cuda_auto_ptr)."""
        site = names.get(r[3], f"0x{r[3]:x}")
        if not r[4]:
            return site
        pair = names.get(r[4], f"0x{r[4]:x}") + "  <- " + site
        # Third frame (depth 3 only) leads: it's the first frame that can belong to the workload.
        if len(r) > 5 and r[5]:
            return names.get(r[5], f"0x{r[5]:x}") + "  <- " + pair
        return pair

    if neg:
        print(f"\nWARNING {len(neg)} site(s) have a NEGATIVE balance and are excluded from the "
              "table, the total and every per-symbol aggregate:")
        for r in sorted(neg, key=lambda r: r[0]):
            print(f"  {r[0]/1e6:14.1f}M  count={r[2]:<8d}  {label_of(r)[:70]}")
        print("  The recorder decremented a site more often than it incremented it, so its free "
              "path attributed a free to the wrong site (cudarec.c note_free). Until that is "
              "fixed, any object whose symbol ALSO owns one of these sites is understated here.")

    total = sum(r[0] for r in rows)
    print(f"\npeak device allocation {total/1e9:.2f} GB across {len(rows)} call sites")
    if meta.get("drops", "0") != "0":
        print(f"  WARNING drops={meta['drops']}: the pointer table filled, so the peak is a "
              "LOWER BOUND. Raise PTR_SLOTS and re-measure rather than reporting this.")
    if meta.get("untracked_frees", "0") != "0":
        print(f"  note untracked_frees={meta['untracked_frees']}: frees of pointers allocated "
              "before the recorder was armed. Harmless unless it is large.")
    if meta.get("stale_reuse", "0") not in ("0", None):
        print(f"  note stale_reuse={meta['stale_reuse']}: addresses that were still live in the "
              "table when the allocator handed them out again, i.e. frees the recorder never saw. "
              "Their bytes were charged back to the site that held them, so the table is correct; "
              "a large count means the interposition set is missing a free entry point.")

    print(f"\n{'device':>12} {'count':>8}  site")
    for r in sorted(rows, key=lambda r: -r[0])[:top]:
        print(f"{r[0]/1e6:11.1f}M {r[2]:8d}  {label_of(r)[:88]}")

    objs = {}
    for r in rows:
        label = label_of(r)
        objs[label] = objs.get(label, 0) + r[0]

    if len(pos) > 2:
        with open(pos[2], "w") as fh:
            fh.write(json.dumps({"rss_kb": total // 1024, "source": "cuda_alloc",
                                 "objects": objs}) + "\n")
        print(f"\nwrote {pos[2]}")


if __name__ == "__main__":
    main()
