"""Diagnostic only (SNNI_PUMA_CENSUS=1): gc census of numpy/jax arrays and (since 2026-08-14) python
bytes at each new RSS maximum, per PUMA process. Bytes are untracked by gc; reached via
gc.get_referents, holder type recorded. Output: census_pid<pid>.txt, one block per new maximum.
"""
import os

if os.environ.get("SNNI_PUMA_CENSUS") == "1":
    import gc
    import sys
    import threading
    import time
    from collections import defaultdict

    _OUT_DIR = os.environ.get("SNNI_PUMA_CENSUS_DIR") or os.environ.get("RESULTS_DIR") or "/tmp"

    def _rss_kb():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except OSError:
            return -1
        return -1

    def _census(out, rss):
        import numpy as np
        groups = defaultdict(lambda: [0, 0])
        seen = set()
        for obj in gc.get_objects():
            try:
                if isinstance(obj, np.ndarray):
                    base = obj.base if obj.base is not None else obj
                    key_id = id(base)
                    if key_id in seen:
                        continue
                    seen.add(key_id)
                    k = ("numpy", str(base.dtype), tuple(base.shape))
                    groups[k][0] += 1
                    groups[k][1] += base.nbytes
            except Exception:
                continue
        try:
            import jax
            for buf in jax.live_arrays():
                try:
                    k = ("jax", str(buf.dtype), tuple(buf.shape))
                    groups[k][0] += 1
                    groups[k][1] += buf.nbytes
                except Exception:
                    continue
        except Exception:
            pass
        # Bytes via a transitive breadth-first walk (they are untracked), holder type kept.
        bytes_groups = defaultdict(lambda: [0, 0])
        try:
            seen_ids = set()
            frontier = gc.get_objects()
            depth = 0
            # Bounded to 8 levels: an unbounded walk on a live interpreter perturbs the run.
            while frontier and depth < 8:
                nxt = []
                for obj in frontier:
                    oid = id(obj)
                    if oid in seen_ids:
                        continue
                    seen_ids.add(oid)
                    try:
                        for ref in gc.get_referents(obj):
                            rid = id(ref)
                            if rid in seen_ids:
                                continue
                            if type(ref) in (bytes, bytearray):
                                seen_ids.add(rid)
                                # Name a frame holder down to the variable (rule 4 needs a last use,
                                # only a named local has one). This branch duplicates the stack scan
                                # below because RunReturn is a generator whose frame is gc-tracked and
                                # reached here first.
                                holder = type(obj).__name__
                                if holder == "frame":
                                    try:
                                        c = obj.f_code
                                        holder = (f"frame {os.path.basename(c.co_filename)}"
                                                  f":{c.co_name}:{c.co_firstlineno}")
                                        for _n, _v in obj.f_locals.items():
                                            if _v is ref:
                                                holder = f"{holder} {_n}"
                                                break
                                    except Exception:
                                        pass
                                k = ("bytes", type(ref).__name__, holder)
                                bytes_groups[k][0] += 1
                                bytes_groups[k][1] += len(ref)
                            else:
                                nxt.append(ref)
                    except Exception:
                        continue
                frontier = nxt
                depth += 1
        except Exception as e:
            print(f"CENSUS|bytes_walk_error|{e!r}", file=out, flush=True)

        # The running stacks, scanned separately and shallowly: gc.get_objects() returns zero frame
        # objects on CPython 3.10, so the walk above cannot see executing locals. Seeding frames into
        # that walk was refuted (+62% peak, +114% wall: a frame reaches the jax/grpc graphs). So each
        # frame gets its own scan, one level into a local container; seen_ids is shared to avoid
        # double counting.
        try:
            frames = []
            skip_ids = set()
            # The census thread skips itself: its own locals reach the buffers it just classified.
            _self = threading.get_ident()
            for _tid, _fr in sys._current_frames().items():
                if _tid == _self:
                    continue
                while _fr is not None and id(_fr) not in skip_ids:
                    skip_ids.add(id(_fr))
                    frames.append(_fr)
                    # Skip a frame's module globals and builtins (not its locals; the seeded walk
                    # exploded on these).
                    try:
                        skip_ids.add(id(_fr.f_globals))
                        skip_ids.add(id(_fr.f_builtins))
                    except Exception:
                        pass
                    _fr = _fr.f_back
            for fr in frames:
                try:
                    c = fr.f_code
                    holder = (f"frame {os.path.basename(c.co_filename)}"
                              f":{c.co_name}:{c.co_firstlineno}")
                except Exception:
                    holder = "frame"
                # Name the local, not just the frame: a frame-only holder is too coarse to choose a
                # free fix from (it once sent a step to delete the wrong local, peak unmoved).
                # f_locals is a snapshot dict built on access, so it is paid for frames only.
                try:
                    items = list(fr.f_locals.items())
                except Exception:
                    items = []
                for name, ref in items:
                    rid = id(ref)
                    if rid in seen_ids or rid in skip_ids:
                        continue
                    if type(ref) in (bytes, bytearray):
                        seen_ids.add(rid)
                        k = ("bytes", type(ref).__name__, f"{holder} {name}")
                        bytes_groups[k][0] += 1
                        bytes_groups[k][1] += len(ref)
                    elif type(ref) in (dict, list, tuple, set, frozenset):
                        for r2 in gc.get_referents(ref):
                            r2id = id(r2)
                            if r2id in seen_ids or r2id in skip_ids:
                                continue
                            if type(r2) in (bytes, bytearray):
                                seen_ids.add(r2id)
                                k = ("bytes", type(r2).__name__, f"{holder} {name}[]")
                                bytes_groups[k][0] += 1
                                bytes_groups[k][1] += len(r2)
        except Exception as e:
            print(f"CENSUS|frame_scan_error|{e!r}", file=out, flush=True)

        total = sum(v[1] for v in groups.values())
        btotal = sum(v[1] for v in bytes_groups.values())
        print(f"CENSUS|newmax|rss_kb={rss}|groups={len(groups)}|array_bytes={total}"
              f"|bytes_groups={len(bytes_groups)}|bytes_total={btotal}", file=out, flush=True)
        for (kind, dt, shape), (n, b) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:25]:
            print(f"  {b:>14d} B  n={n:<5d} {kind:<6s} {dt:<10s} {shape}", file=out, flush=True)
        for (kind, dt, holder), (n, b) in sorted(bytes_groups.items(), key=lambda kv: -kv[1][1])[:25]:
            print(f"  {b:>14d} B  n={n:<5d} {kind:<6s} {dt:<10s} held_by={holder}",
                  file=out, flush=True)

    def _loop():
        pid = os.getpid()
        path = os.path.join(_OUT_DIR, f"census_pid{pid}.txt")
        out = open(path, "w", buffering=1)
        print(f"LEVER|snni_puma_census|armed pid={pid}|walks=numpy,jax,bytes", file=out, flush=True)
        best = 0
        while True:
            time.sleep(0.5)
            rss = _rss_kb()
            if rss > best * 1.02 or (rss > best and rss - best > 131072):
                best = max(best, rss)
                try:
                    _census(out, rss)
                except Exception as e:
                    print(f"CENSUS|error|{e!r}", file=out, flush=True)

    threading.Thread(target=_loop, daemon=True, name="snni_census").start()
