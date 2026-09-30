#!/usr/bin/env python3
"""Device peak decomposition from a torch memory snapshot -> objects_vram.jsonl (make_steps.py input).

Modes:
  state (default)  live blocks with allocation stacks; snapshot taken at a new allocation maximum
                   by tools/attrib_gpu/sitecustomize.py, so the live set is the peak.
  all (fallback)   adds `device_traces` (full alloc/free history), replayed to the peak;
                   cost +625 MB host RSS, +14% wall.
Both agree on SHAFT-GPU s0 to 0.1 MB (top four: 2850.4 / 866.6 / 713.0 / 475.1 MB).

Usage: torch_peak_objects.py <snapshot.pickle> [objects_vram.jsonl]
"""
import json
import pickle
import sys
from collections import defaultdict


def frame_label(frames, depth=3):
    """Name an allocation by its innermost frames, skipping allocator plumbing."""
    if not frames:
        return "<no stack>"
    picked = []
    for f in frames:
        fn = f.get("filename", "")
        name = f.get("name", "?")
        if "torch/cuda/memory" in fn or "_record_memory_history" in name:
            continue
        picked.append(f"{fn.rsplit('/', 1)[-1]}:{f.get('line', '?')}:{name}")
        if len(picked) >= depth:
            break
    return " <- ".join(picked) if picked else "<no stack>"


def from_state(snap):
    """Live blocks in the snapshot; used when the watchdog captured it at the peak."""
    agg, total = defaultdict(int), 0
    for seg in snap.get("segments", []):
        for b in seg.get("blocks", []):
            if b.get("state") != "active_allocated":
                continue
            total += b["size"]
            agg[frame_label(b.get("frames"))] += b["size"]
    return total, dict(agg)


def from_history(snap):
    """Replay alloc/free events to the moment live allocation was highest (all mode only)."""
    best_peak, best_live = 0, {}
    for events in (snap.get("device_traces") or []):
        live, cur, peak, peak_live = {}, 0, 0, {}
        for ev in events:
            act, size, addr = ev.get("action"), ev.get("size", 0), ev.get("addr")
            if act == "alloc":
                live[addr] = (size, frame_label(ev.get("frames")))
                cur += size
                if cur > peak:
                    peak, peak_live = cur, dict(live)
            elif act in ("free_completed", "free_requested") and addr in live:
                cur -= live[addr][0]
                del live[addr]
        if peak > best_peak:
            agg = defaultdict(int)
            for _a, (size, label) in peak_live.items():
                agg[label] += size
            best_peak, best_live = peak, dict(agg)
    return best_peak, best_live


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    snap = pickle.load(open(sys.argv[1], "rb"))

    total, objs = from_state(snap)
    source = "torch_state"
    # Fall back to replay only if the snapshot holds no live blocks but does hold a history --
    # i.e. it was taken at exit in `all` mode rather than at the peak by the watchdog.
    if total == 0 and (snap.get("device_traces") or []):
        total, objs = from_history(snap)
        source = "torch_history"
    if not total:
        sys.exit("snapshot holds neither live blocks nor an event history")

    print(f"peak live = {total/1e6:.0f} MB across {len(objs)} allocation sites  [{source}]")
    for label, size in sorted(objs.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {size/1e6:9.1f} MB  {100*size/total:5.1f}%  {label}")

    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as fh:
            fh.write(json.dumps({"rss_kb": total // 1024, "source": source,
                                 "objects": objs}) + "\n")


if __name__ == "__main__":
    main()
