"""Diagnostic only. Names Python-owned objects the C recorder collapses onto c10::alloc_cpu. Separate
run; peak value from the probe-free run; cross-checked. Groups by owner (CrypTen module
_parameters/_buffers by data_ptr) and by (shape, dtype). Never in a measured run (SHAFT_LEVERS, RUN_META).
"""

import gc
import os
import sys
import threading
import time

import torch

_OUT = None


def _rss_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return -1


def _storage_ptr(t):
    st = t.untyped_storage() if hasattr(t, "untyped_storage") else t.storage()
    nb = st.nbytes() if callable(getattr(st, "nbytes", None)) else st.size() * t.element_size()
    return st.data_ptr(), nb


def _owners(objs):
    """data_ptr -> owner label, for every tensor a CrypTen module holds. Imports lazily and fails
    soft (a probe must never take the run down). Label is the module slot, not class."""
    try:
        from crypten.nn import module as cnnmod
    except Exception:
        return {}, 0
    owned, mods = {}, 0
    for obj in objs:
        if not isinstance(obj, cnnmod.Module):
            continue
        mods += 1
        for slot, label in (("_parameters", "graph_param"), ("_buffers", "graph_buffer")):
            for _, t in list(getattr(obj, slot, {}).items()):
                s = t if torch.is_tensor(t) else getattr(t, "share", None)
                if not torch.is_tensor(s):
                    continue
                try:
                    ptr, _ = _storage_ptr(s)
                except Exception:
                    continue
                owned[ptr] = label
    # Graph.forward's `values` dict is an owner this join cannot reach (it is a local of forward, not
    # an attribute). It matters once a lever moves shares out of the modules: they reappear as
    # anonymous rows, and levers.env then has to argue from shapes plus graph_param going to zero.
    return owned, mods


def _census(tag, rss):
    """Live torch tensors, deduplicated by data_ptr (a view and its base share one allocation;
    counting both would inflate past the resident peak), grouped by (shape,dtype) AND by owner."""
    objs = gc.get_objects()
    seen = {}
    for obj in objs:
        try:
            if not isinstance(obj, torch.Tensor):
                continue
            ptr, nb = _storage_ptr(obj)
        except Exception:
            continue
        if ptr in seen:
            continue
        seen[ptr] = (nb, tuple(obj.shape), str(obj.dtype))

    groups = {}
    for nb, shape, dt in seen.values():
        k = (shape, dt)
        n, tot = groups.get(k, (0, 0))
        groups[k] = (n + 1, tot + nb)

    total = sum(t for _, t in groups.values())
    print(f"CENSUS|{tag}|rss_kb={rss}|live_tensor_bytes={total}|groups={len(groups)}|"
          f"allocations={len(seen)}", file=_OUT, flush=True)
    for (shape, dt), (n, tot) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:25]:
        print(f"  {tot:>13d} B  n={n:<4d} {dt:<14s} {shape}", file=_OUT, flush=True)

    # The ranking: owned tensors group by owner (one object, one row); unowned keep (shape,dtype).
    owned, mods = _owners(objs)
    ranked = {}
    for ptr, (nb, shape, dt) in seen.items():
        k = owned.get(ptr) or f"unowned {dt} {shape}"
        n, tot = ranked.get(k, (0, 0))
        ranked[k] = (n + 1, tot + nb)
    print(f"CENSUS_OWNER|{tag}|rss_kb={rss}|owner_groups={len(ranked)}|"
          f"crypten_modules={mods}|owned_allocations={len(owned)}", file=_OUT, flush=True)
    for k, (n, tot) in sorted(ranked.items(), key=lambda kv: -kv[1][1])[:25]:
        print(f"  {tot:>13d} B  n={n:<4d} {k}", file=_OUT, flush=True)


def _watch(interval_s=0.05):
    best = 0
    while True:
        rss = _rss_kb()
        # 1% above the previous max: census every sample would spend the run in the gc walk; a
        # threshold keeps them on the way UP the peak, where the objects that build it are.
        if rss > best * 1.01:
            best = rss
            try:
                _census("newmax", rss)
            except Exception as exc:  # a probe must never take the run down with it
                print(f"CENSUS|error|{exc}", file=_OUT, flush=True)
        time.sleep(interval_s)


def _start():
    global _OUT
    # Keyed by pid, not rank: imported at module scope before the launcher sets RANK, so every party
    # would otherwise write the same file. party<N>.pid recovers the party identity afterwards.
    pid = os.getpid()
    d = os.environ.get("RESULTS_DIR", "/tmp")
    _OUT = open(os.path.join(d, f"census_pid{pid}.txt"), "w", buffering=1)
    print(f"LEVER|probe_census_cpu|patched pid={pid}", file=sys.stderr, flush=True)
    t = threading.Thread(target=_watch, daemon=True)
    t.start()


_start()
