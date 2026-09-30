#!/usr/bin/env python3
"""Assembles the canonical host decomposition from three sources:

    heap:<site>   heap objects by allocation site, RESIDENT bytes (--heap, from resolve_resident.py)
    file:<lib>    file-backed mappings by library path            (smaps, here) = the measured
                  library-overhead term make_steps.py subtracts from the peak; kept prefixed
    anon:residual anonymous memory the heap profiler did not attribute (arenas, CUDA host, stacks)

--peak-kb records the snapshot RSS against the true peak and warns on divergence.

Usage: smaps_objects.py <peaks-dir-or-smaps-file> [objects_host.jsonl] [--heap objects_heap.jsonl]
                        [--peak-kb N]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict


def pick_snapshot(path):
    """The snapshot at the highest RSS, not the newest -- later ones can be past the peak."""
    if os.path.isfile(path):
        return path
    # poll_peak.sh rewrites peak.smaps at every new maximum, so it is the peak by construction; the
    # thresholded timeline is a fallback for older traces and can sit well below the peak.
    exact = os.path.join(path, "peak.smaps")
    if os.path.exists(exact):
        return exact
    idx = os.path.join(path, "INDEX")
    best, best_rss = None, -1
    if os.path.exists(idx):
        for ln in open(idx, errors="replace"):
            if ln.startswith("#"):
                continue
            p = ln.split()
            if len(p) >= 2 and p[1].isdigit() and int(p[1]) > best_rss:
                cand = os.path.join(path, f"{p[0]}.smaps")
                if os.path.exists(cand):
                    best, best_rss = cand, int(p[1])
    if best:
        return best
    snaps = sorted(glob.glob(os.path.join(path, "*.smaps")))
    return snaps[-1] if snaps else None


def parse(path):
    """Return (file_backed {path: kB}, anon_kB, total_kB)."""
    files, anon, total = defaultdict(int), 0, 0
    name, is_file = None, False
    with open(path, errors="replace") as fh:
        for ln in fh:
            p = ln.split()
            if not p:
                continue
            if "-" in p[0] and p[0][0].isalnum() and ":" in ln[:80]:
                # mapping header: addr perms offset dev inode [path]. Pseudo-mappings ([heap], etc.)
                # have a name but are not file-backed; counting them as libraries misattributes heap.
                nm = p[-1] if len(p) >= 6 else None
                is_file = bool(nm) and not nm.startswith("[")
                name = nm if is_file else None
            elif ln.startswith("Rss:") and len(p) >= 2:
                rss = int(p[1])
                total += rss
                if is_file and name:
                    files[name] += rss
                else:
                    anon += rss
    return dict(files), anon, total


def load_heap(path):
    """Heap objects from a heap profiler or resolve_resident.py -> ({site: bytes}, source, meta).

    meta carries the producer's provenance (sample RSS, drops), which make_steps.py publishes.
    """
    best, best_tot, src, meta = {}, -1, "heap", {}
    with open(path, errors="replace") as fh:
        for ln in fh:
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            objs = rec.get("objects", {})
            tot = sum(objs.values())
            if tot > best_tot:
                best, best_tot, src = objs, tot, rec.get("source", "heap")
                meta = {k: rec[k] for k in ("sample_rss_kb", "sample_ts_ms", "drops",
                                            "maps", "fullscan_skipped_bytes", "fullscan_mode",
                                            "shapes")
                        if k in rec}
    return best, src, meta


def locate_residual(maps):
    """Turn the fullscan's per-mapping table into named anonymous residual entries.

    Only ANONYMOUS mappings (a file-backed mapping's unaccounted bytes are already the file: term).
    Names stay under anon: (locating is not attributing; still not attackable targets), but separate
    the three findings a single residual number hides: [heap] retention, a direct mmap, sub-floor allocs.
    """
    heap_b = stack_b = 0
    other = []
    for m in maps:
        un = m.get("unaccounted", 0)
        if un <= 0:
            continue
        path = m.get("path", "")
        if path.startswith("[heap]"):
            heap_b += un
        elif path.startswith("[stack"):
            stack_b += un
        elif not path:
            other.append(un)
    out = {}
    if heap_b:
        out["anon:[heap]"] = heap_b
    if stack_b:
        out["anon:[stack]"] = stack_b
    # Largest anonymous mappings get their own rows ("one 500 MB mapping" vs "500 small ones" are
    # different answers); cut at 1% of the located total, tail summed into one counted row.
    other.sort(reverse=True)
    cut = 0.01 * (heap_b + stack_b + sum(other))
    big = [b for b in other if b >= cut]
    for i, b in enumerate(big, 1):
        out[f"anon:mmap-{i}"] = b
    tail = other[len(big):]
    if tail:
        out[f"anon:mmap-rest({len(tail)})"] = sum(tail)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = pick_snapshot(sys.argv[1])
    if not src:
        sys.exit(f"no smaps snapshot under {sys.argv[1]}")
    heap, heap_src, heap_meta = {}, "heap", {}
    if "--heap" in sys.argv:
        heap, heap_src, heap_meta = load_heap(sys.argv[sys.argv.index("--heap") + 1])
    heap_kb = sum(heap.values()) // 1024
    peak_kb = int(sys.argv[sys.argv.index("--peak-kb") + 1]) if "--peak-kb" in sys.argv else 0
    # Opt-in: --retention-row <name> replaces the located anon: entries with one named aggregate,
    # honest only where a second producer has shown what those bytes are. Default off.
    retention_row = (sys.argv[sys.argv.index("--retention-row") + 1]
                     if "--retention-row" in sys.argv else "")
    if not peak_kb:
        # The poller writes the true maximum next to the snapshot, so this needs no argument in
        # the normal case.
        pk = os.path.join(sys.argv[1], "PEAK")
        if os.path.isdir(sys.argv[1]) and os.path.exists(pk):
            parts = open(pk, errors="replace").read().split()
            if len(parts) >= 2 and parts[1].isdigit():
                peak_kb = int(parts[1])

    # The reference peak is VmHWM, not the polled max: the process runs between the RSS sample and
    # the stop, so a valid frozen snapshot sits slightly above the polled value (d6_ars_free: 668916
    # vs polled 661400, but 2.1% below the kernel's 683016). A snapshot above VmHWM is still an error.
    hwm_kb = 0
    if os.path.isdir(sys.argv[1]):
        vm = os.path.join(os.path.dirname(sys.argv[1].rstrip("/")), "vmrss.log")
        if not os.path.exists(vm):
            vm = os.path.join(sys.argv[1], "vmrss.log")
        if os.path.exists(vm):
            with open(vm, errors="replace") as fh:
                for ln in fh:
                    if ln.startswith("#"):
                        m = re.search(r"hwm_kb=([0-9]+)", ln)
                        if m:
                            hwm_kb = max(hwm_kb, int(m.group(1)))
                        continue
                    p = ln.split()
                    if len(p) >= 4 and p[3].isdigit():
                        hwm_kb = max(hwm_kb, int(p[3]))
    polled_kb = peak_kb
    if hwm_kb:
        peak_kb = max(peak_kb, hwm_kb)

    misaligned = False
    files, anon, total = parse(src)
    # Heap profiler and poller are separate runs, so the heap total can exceed this anonymous total;
    # clamp to zero (the overshoot shows in the printed reconciliation).
    residual = max(0, anon - heap_kb)

    # With the full scan the residual is MEASURED per mapping; the subtraction is kept beside it as
    # the standing two-mechanisms check (a disagreement is an instrument finding).
    located = locate_residual(heap_meta.get("maps", []))
    located_kb = sum(located.values()) // 1024

    print(f"snapshot: {src}")
    print(f"  total RSS      {total/1024:9.1f} MB")
    if peak_kb:
        frac = total / peak_kb
        print(f"  run peak       {peak_kb/1024:9.1f} MB   snapshot is {100*frac:.1f}% of it")
        # A snapshot above the recorded peak is as wrong as one far below it (different instants:
        # d6_ars_free gave 103% attribution and a long-removed object back at 650 MB). Cut at 1.005,
        # not 1.02: valid snapshots report frac=1.0000, and 1.02 let through the run that motivated it.
        if frac > 1.005:
            print(f"  ERROR: the composition snapshot ({total} kB) is ABOVE the recorded peak "
                  f"({peak_kb} kB). These are different instants; the decomposition is not "
                  f"describing the published peak and must not be used.")
            # Marked in the record, not merely printed (d7_sumsq_free had all three replicates this
            # way, attributing 102-104%, one copied into the published dir before anyone checked).
            misaligned = True
        if frac < 0.9:
            print("  WARNING: the composition snapshot is well below the peak, so this peak is a "
                  "TRANSIENT. The library term is still usable (file-backed mappings are stable), "
                  "but the snapshot does not describe the peak instant. The heap term is "
                  "unaffected -- it is reconstructed in-process at the heap's own maximum.")
    print(f"  file-backed    {sum(files.values())/1024:9.1f} MB  ({100*sum(files.values())/total:.1f}%)"
          if total else "")
    print(f"  anonymous      {anon/1024:9.1f} MB  ({100*anon/total:.1f}%)" if total else "")
    if heap_kb:
        print(f"    of which heap-attributed {heap_kb/1024:.1f} MB "
              f"({len(heap)} sites), residual {residual/1024:.1f} MB")
    if located:
        gap = located_kb - residual
        print(f"    residual LOCATED {located_kb/1024:.1f} MB across {len(located)} entries "
              f"(subtraction said {residual/1024:.1f} MB, difference {gap/1024:+.1f} MB)")
        for k, v in sorted(located.items(), key=lambda kv: -kv[1]):
            print(f"     {v/1e6:9.1f} MB  {k}")
        if residual and abs(gap) > 0.05 * residual:
            print("    NOTE: the two mechanisms disagree by more than 5%. They are taken at "
                  "different instants, so a moving peak explains some of it; a large or growing "
                  "gap is about the instrument and belongs resolved before this table is used.")
        if heap_meta.get("fullscan_skipped_bytes"):
            print(f"    ({heap_meta['fullscan_skipped_bytes']/1e9:.1f} GB of PROT_NONE "
                  f"reservation was not scanned; measured to hold nothing, verify with "
                  f"SNNI_PM_FULLSCAN=all)")
    print("  largest file-backed mappings:")
    for name, kb in sorted(files.items(), key=lambda kv: -kv[1])[:10]:
        print(f"   {kb/1024:9.1f} MB  {100*kb/total:5.1f}%  {os.path.basename(name)}")

    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        objs = {f"heap:{k}": v for k, v in heap.items()}
        # The recorder tables are file-backed but are the instrument, so they get an instrument:
        # prefix (visible, excluded from file: so not double-subtracted; make_steps.py takes them off
        # the peak). Both recorders name their table: pmtable_* (host) and devtable_* (device, which
        # was anonymous calloc until it became named, +1.02% of MOAI-GPU's host peak).
        _INSTR = ("pmtable", "devtable")
        for k, v in files.items():
            b = os.path.basename(k)
            key = f"instrument:{b}" if b.startswith(_INSTR) else f"file:{b}"
            objs[key] = objs.get(key, 0) + v * 1024
        if located and retention_row:
            # A name on the located bytes, valid only because a second producer over the same freeze
            # identified them as allocator retention (not the mmap/sub-floor alternatives locate_residual
            # cannot distinguish). The per-mapping breakdown is still printed under "residual LOCATED".
            objs[retention_row] = sum(located.values())
        elif located:
            # Measured per mapping; anon:residual is omitted here (a subtraction-derived row beside
            # measured ones would double-count and read as a second finding).
            objs.update(located)
        elif residual:
            objs["anon:residual"] = residual * 1024
        source = f"{heap_src}+smaps" if heap else "smaps"
        # A snapshot at a different instant describes no single moment, so the source says so and
        # make_steps.py clears the object columns.
        if misaligned:
            source = "MISALIGNED"
        rec = {"rss_kb": total, "source": source,
               "snapshot_rss_kb": total, "run_peak_kb": peak_kb,
               # Both references travel: the polled max triggered the snapshot, VmHWM is the published
               # peak. Keeping only one makes a kernel-valid capture look wrong against the poller.
               "run_polled_peak_kb": polled_kb, "run_hwm_kb": hwm_kb or None,
               "snapshot_frac_of_peak": round(total / peak_kb, 4) if peak_kb else None,
               "objects": objs}
        # The per-mapping table stays in objects_resident.jsonl; only its provenance travels here.
        rec.update({k: v for k, v in heap_meta.items() if k != "maps"})
        if located:
            rec["residual_source"] = "fullscan"
            rec["residual_subtraction_kb"] = residual
        with open(sys.argv[2], "w") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"wrote {sys.argv[2]}  [{source}]")


if __name__ == "__main__":
    main()
