"""Peak-time GPU object attribution via sitecustomize (PYTHONPATH); enable with SNNI_ATTRIB=1.

SNNI_ATTRIB_MODE=state (default): watchdog snapshot at each new allocation high, live-block stacks.
SNNI_ATTRIB_MODE=all: also full alloc/free history for replay (+625 MB host RSS, +14% wall), fallback.
"""
import atexit
import os
import sys
import threading

if os.environ.get("SNNI_ATTRIB") == "1":
    try:
        import torch

        _MODE = os.environ.get("SNNI_ATTRIB_MODE", "state")
        _MAX = int(os.environ.get("SNNI_ATTRIB_ENTRIES", "4000000"))
        _POLL_S = float(os.environ.get("SNNI_ATTRIB_POLL_MS", "50")) / 1000.0
        _TRIGGER = float(os.environ.get("SNNI_ATTRIB_TRIGGER_PCT", "2")) / 100.0

        def _arm():
            if _MODE == "all":
                torch.cuda.memory._record_memory_history(
                    enabled="all", context="all", stacks="python", max_entries=_MAX
                )
            else:
                # No event history: stacks are attached to live blocks only.
                torch.cuda.memory._record_memory_history(
                    enabled="state", context="state", stacks="python"
                )

        _arm()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=_arm)

        _best = {"alloc": 0, "snap": None}
        _lock = threading.Lock()

        def _watch():
            """Snapshot when device allocation reaches a new high. `state` mode keeps no history,
            so this is what catches the peak rather than only the tail at exit; only a materially
            higher maximum triggers a snapshot, to avoid dumping continuously during warm-up."""
            while True:
                try:
                    cur = torch.cuda.memory_allocated()
                except Exception:
                    return
                if cur > _best["alloc"] * (1.0 + _TRIGGER):
                    try:
                        snap = torch.cuda.memory._snapshot()
                    except Exception:
                        snap = None
                    if snap is not None:
                        with _lock:
                            if cur > _best["alloc"]:
                                _best["alloc"], _best["snap"] = cur, snap
                threading.Event().wait(_POLL_S)

        if _MODE != "all":
            t = threading.Thread(target=_watch, name="snni-attrib", daemon=True)
            t.start()

        def _dump():
            try:
                import pickle

                out_dir = os.environ.get("RESULTS_DIR") or "/results"
                rank = os.environ.get("RANK", "0")
                pid = os.getpid()
                if _MODE == "all":
                    snap = torch.cuda.memory._snapshot()
                    peak_alloc = None
                else:
                    with _lock:
                        snap, peak_alloc = _best["snap"], _best["alloc"]
                # The orchestrating parent also runs this hook; pid in the filename keeps it
                # from overwriting a real party's dump with an empty one.
                if snap is None or (
                    not snap.get("segments") and not any(snap.get("device_traces") or [])
                ):
                    sys.stderr.write(f"SNNI_ATTRIB|rank={rank}|pid={pid}|empty,skipped\n")
                    return
                path = os.path.join(out_dir, f"cuda_memhistory_rank{rank}_pid{pid}.pickle")
                with open(path, "wb") as fh:
                    pickle.dump(snap, fh)
                extra = f"|peak_alloc_at_snapshot={peak_alloc}" if peak_alloc else ""
                sys.stderr.write(
                    f"SNNI_ATTRIB|rank={rank}|mode={_MODE}|wrote={path}{extra}\n"
                )
            except Exception as exc:  # never let attribution break the run it observes
                sys.stderr.write(f"SNNI_ATTRIB|dump_failed={exc!r}\n")

        atexit.register(_dump)
        sys.stderr.write(f"SNNI_ATTRIB|recording=on|mode={_MODE}\n")
    except Exception as exc:
        sys.stderr.write(f"SNNI_ATTRIB|init_failed={exc!r}\n")
