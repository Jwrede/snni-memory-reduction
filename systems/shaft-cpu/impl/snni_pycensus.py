"""In-run object table for a Python system.

    import snni_pycensus; snni_pycensus.arm_signal()      # once, at interpreter start
    # the poller outside then sends SIGUSR1 at each new high-water

Resident bytes per Python-owned object (numpy/ATen allocations owned by Python objects are invisible to
the C recorder). Transitive gc.get_referents walk reaches untracked numpy arrays and bytes (316,354
objects in 982 ms; tracemalloc was 8x slower). Main thread, GIL held, residency from /proc/self/pagemap
for the enumerated ranges (one instant, no external freeze). Signal-driven (safe point between
bytecodes). Covers Python-owned memory only; the C recorder covers the rest, reported side by side.
Row: PYOBJ <resident bytes> <size bytes> <kind> <detail> held_by=<owner>; held_by = the name a step
targets (rule 4 grouping).
"""
import gc
import os
import sys

_ENABLED = os.environ.get("SNNI_PY_CENSUS") == "1"
_OUT_DIR = os.environ.get("SNNI_PY_CENSUS_DIR") or "/tmp"
# Same 64 KiB floor as the C recorder, so the two tables answer the same question from independent
# directions; a higher floor hides objects and cannot be compared (recorder 2766.5 MB/504 allocs vs
# a 1 MiB census 1717.4 MB/241 at the frozen peak).
_MIN = int(os.environ.get("SNNI_PY_CENSUS_MIN", str(1 << 16)))
_PAGE = os.sysconf("SC_PAGE_SIZE")
# Bit 63 of a pagemap entry (high bit of its eighth byte) is `present`; this maps that byte to 1/0
# so counting is one C call instead of a per-page loop.
_PRESENT_TABLE = bytes(1 if (i & 0x80) else 0 for i in range(256))
_CONTAINERS = (list, tuple, dict, set, frozenset)


def _rss_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def _resident(fh, addr, size):
    """Bytes of [addr, addr+size) the kernel has actually backed, or -1. Asked from inside so it
    lands in the same instant as the walk; first and last pages counted whole (external sampler's
    convention), error bounded by two pages per object."""
    if fh is None or not addr or size <= 0:
        return -1
    first = addr // _PAGE
    n = ((addr + size - 1) // _PAGE) - first + 1
    try:
        fh.seek(first * 8)
        buf = fh.read(n * 8)
    except (OSError, ValueError, OverflowError):
        return -1
    if len(buf) < n * 8:
        return -1
    # Counted in C, not a Python loop: an 866 MB object is 211k pages, and counting them one at a
    # time made a census cost seconds and the run +49% wall. Slice/translate/count are all C-level.
    return _PRESENT_TABLE and buf[7::8].translate(_PRESENT_TABLE).count(1) * _PAGE


def _shape(shape):
    """`(28996,768)` and never `(28996, 768)`: a row is read field by field, so a space inside one
    field shifts every field after it (the first parse put the holder in the shape's column)."""
    return "(" + ",".join(str(int(x)) for x in shape) + ")"


_TENSOR_TYPES = {}


def _is_tensor(t, torch):
    """Is `t` a tensor type, subclasses included, WITHOUT running user code per object (isinstance
    consults a metaclass hook). Exact typing missed torch.nn.Parameter (a Tensor subclass, and a
    model's weights are Parameters: 980 MB decoding to the plaintext BERT weights). Checked once per
    type and cached, so issubclass runs a few dozen times over a census, not half a million."""
    k = _TENSOR_TYPES.get(t)
    if k is None:
        try:
            k = issubclass(t, torch.Tensor)
        except Exception:
            k = False
        _TENSOR_TYPES[t] = k
    return k


def _describe(parent):
    """A short, safe name for the object that held a leaf: type name only, so nothing calls user
    code (type() cannot run __getattr__ or a property)."""
    try:
        return type(parent).__name__
    except Exception:
        return "?"


def dump(phase):
    if not _ENABLED:
        return
    try:
        _dump(phase)
    except Exception as exc:            # an instrument must never take the run with it
        try:
            with open(os.path.join(_OUT_DIR, f"pycensus_pid{os.getpid()}.txt"), "a") as fh:
                print(f"PYCENSUS|error|phase={phase}|{exc!r}", file=fh, flush=True)
        except OSError:
            pass


def _dump(phase):
    np = sys.modules.get("numpy")
    torch = sys.modules.get("torch")

    rows = []
    seen = set()        # leaves already recorded, so a shared buffer is not listed twice
    walked = set()      # containers already expanded, which is a different question
    visited = 0

    def note(obj, size, kind, detail, parent):
        if size < _MIN:
            return
        a = id(obj)
        if a in seen:
            return
        seen.add(a)
        rows.append([size, kind, detail, _describe(parent), obj])

    def leaf(r, parent):
        """Record r if it is one of the kinds this system's memory is made of. Exact types only."""
        rt = type(r)
        try:
            if np is not None and rt is np.ndarray:
                note(r, r.nbytes, "numpy", f"{r.dtype}{_shape(r.shape)}", parent)
            elif rt is bytes or rt is bytearray:
                note(r, len(r), rt.__name__, "-", parent)
            elif torch is not None and _is_tensor(rt, torch):
                note(r, r.untyped_storage().nbytes(), "torch", f"{r.dtype}{_shape(r.shape)}", parent)
            else:
                return False
        except Exception:
            pass
        return True

    # Roots are swept last: a tensor is tracked, so it appears in gc.get_objects() itself; recording
    # it there marks it seen and the walk never reaches it through its holder (every such row would
    # read held_by=NoneType). Walking referents first gives each leaf the parent it was reached
    # through; the root sweep afterwards picks up only what nothing tracked points at.
    roots = gc.get_objects()
    stack = list(roots)
    while stack:
        batch, stack = stack, []
        for o in batch:
            i = id(o)
            if i in walked:
                continue
            walked.add(i)
            visited += 1
            try:
                refs = gc.get_referents(o)
            except Exception:
                continue
            for r in refs:
                if leaf(r, o):
                    continue
                if id(r) not in walked:
                    stack.append(r)

    for o in roots:
        leaf(o, None)
    del roots

    try:
        pm = open("/proc/self/pagemap", "rb", buffering=0)
    except OSError:
        pm = None

    # One buffer, one row: two Python objects can view the SAME bytes (the spill lever does
    # torch.from_numpy(arr)), and counted twice the census totalled 2.34 GB inside a 1.72 GB process.
    # Both rows are kept (a reader wants the aliasing); only the second stops contributing to the total.
    rows.sort(key=lambda r: -r[0])
    counted = set()
    total_res = 0
    for row in rows:
        size, kind, _detail, _held, obj = row
        try:
            if kind == "numpy":
                addr = obj.__array_interface__["data"][0]
            elif kind == "torch":
                addr = obj.untyped_storage().data_ptr()
            else:
                addr = id(obj)
        except Exception:
            addr = 0
        res = max(_resident(pm, addr, size), 0)
        alias = addr in counted and addr != 0
        if not alias:
            counted.add(addr)
            total_res += res
        row.append(res)
        row.append("alias" if alias else "-")
        row.append(addr)
    if pm is not None:
        pm.close()

    # What the recorder sees and this census does not, measured in the same instant and process. The
    # C recorder's live (pointer,size,site) table is read HERE because that is the only way to compare
    # the two instruments at ONE moment (its FILE holds what was live at process EXIT: 360 allocs /
    # 656.7 MB vs the recorder's own 2777.1 MB peak snapshot). An allocation is NAMED when a census
    # object lies inside it; what is left bounds what a Python-level census can ever explain.
    unmatched_n = unmatched_b = matched_n = matched_b = 0
    unmatched_sz = {}
    try:
        import bisect
        import struct
        tpath = os.environ.get("SNNI_PM_TABLE", "")
        if "%d" in tpath:
            tpath = tpath.replace("%d", str(os.getpid()))
        if tpath and os.path.exists(tpath):
            with open(tpath, "rb") as th:
                magic, _ins, _drops, nslots = struct.unpack("<QQQQ", th.read(32))
                blob = th.read(nslots * 32) if magic == 0x504D52454332 else b""
            keys = sorted(a for a in counted if a)
            for off in range(0, len(blob), 32):
                ptr, sz = struct.unpack_from("<QQ", blob, off)
                if not ptr:
                    continue
                i = bisect.bisect_left(keys, ptr)
                if i < len(keys) and keys[i] < ptr + sz:
                    matched_n += 1
                    matched_b += sz
                else:
                    unmatched_n += 1
                    unmatched_b += sz
                    unmatched_sz[sz] = unmatched_sz.get(sz, 0) + 1
    except Exception:
        pass

    path = os.path.join(_OUT_DIR, f"pycensus_pid{os.getpid()}.txt")
    with open(path, "a") as fh:
        rss = _rss_kb()
        # The instrument checks itself: resident object bytes cannot exceed the process's own
        # resident size; if they do, something is double-counted, announced here as OVER_RSS.
        over = "" if rss <= 0 or total_res <= rss * 1024 else "|OVER_RSS"
        print(f"PYCENSUS|phase={phase}|rss_kb={rss}|objects={len(rows)}"
              f"|visited={visited}|size_bytes={sum(r[0] for r in rows)}"
              f"|resident_bytes={total_res}"
              f"|covered={0 if rss <= 0 else round(100.0 * total_res / (rss * 1024), 1)}%{over}"
              f"|rec_matched={matched_n}/{matched_b}|rec_unmatched={unmatched_n}/{unmatched_b}",
              file=fh, flush=True)
        # The address is printed so this table can be JOINED to the C recorder's (the coverage check:
        # recorder 2777.1 MB at the frozen peak, this census reaches 1361.4 MB, so ~1.4 GB is held by
        # something that is not a Python object). The unmatched size histogram says WHAT is missing
        # (a protocol's buffers come in a few exact sizes): a standing pool of ~267 allocs / ~980 MB.
        for sz, n in sorted(unmatched_sz.items(), key=lambda kv: -kv[0] * kv[1])[:12]:
            print(f"RECGAP {sz * n} {n} {sz}", file=fh, flush=True)
        for size, kind, detail, held, _obj, res, alias, addr in rows:
            print(f"PYOBJ {res} {size} {kind} {detail} held_by={held} {alias} 0x{addr:x}",
                  file=fh, flush=True)


_BUSY = [False]
_LAST = [0.0, 0]          # monotonic seconds, rss_kb at the last census that ran
# The census decides when a signal is worth acting on (the poller signals freely); the policy lives
# here because only this side knows a census costs about a second against a 15 ms poll interval.
_MIN_GAP_S = float(os.environ.get("SNNI_PY_CENSUS_MIN_GAP_S", "3"))
_MIN_GROW = float(os.environ.get("SNNI_PY_CENSUS_MIN_GROW", "1.08"))


def _on_signal(_sig, _frame):
    """Answer a poller signal, or decline it, and never re-enter. The guard: CPython runs a handler
    between bytecodes INCLUDING while another handler runs, so a second signal during a census starts
    a second on top (175 signals over 8 min once meant no census ever finished). So one census at a
    time, and only when the process has grown since the last."""
    import time
    if _BUSY[0]:
        return
    now = time.monotonic()
    rss = _rss_kb()
    if _LAST[0] and (now - _LAST[0] < _MIN_GAP_S or rss < _LAST[1] * _MIN_GROW):
        return
    _BUSY[0] = True
    try:
        dump("signal")
        _LAST[0], _LAST[1] = time.monotonic(), rss
    finally:
        _BUSY[0] = False


def arm_signal(sig=None):
    """Answer SIGUSR1 with a census. Must be called from the main thread. Also sets the signal to
    IGNORE at exit: SIGUSR1's default action is to KILL, and during teardown the Python callback can
    no longer run, so a late signal would kill a program that had already written its gate (job
    46009906, exit code -10)."""
    if not _ENABLED:
        return
    import atexit
    import signal
    s = sig if sig is not None else signal.SIGUSR1
    try:
        signal.signal(s, _on_signal)
        atexit.register(lambda: signal.signal(s, signal.SIG_IGN))
    except (ValueError, OSError):
        pass
