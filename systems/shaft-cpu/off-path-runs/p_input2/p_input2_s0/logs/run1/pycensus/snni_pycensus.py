"""The object table for a Python system, taken inside the measured run itself.

    import snni_pycensus; snni_pycensus.arm_signal()      # once, at interpreter start
    # the poller outside then sends SIGUSR1 at each new high-water

WHAT IT IS FOR. This campaign decomposes each peak into the objects that make it up, in RESIDENT
bytes, ranked. On twelve of thirteen systems the C recorder does all of it: it records
`(address, size, allocating function)` and a sampler reads page residency for each range. On this
one it cannot finish the job, and `p_depth3` and `p_depth4` measured why rather than assuming it:
the program is Python, so the native stack runs numpy or ATen straight into the interpreter and
never passes through the measured code. At every depth tried, both frames stayed inside the same
third-party library. The owner is a Python object, so the producer has to be a Python one.

## Three things had to be true, and each was measured before it was believed

**1. The collector alone is blind.** A walk of `gc.get_objects()` finds torch tensors and NOT ONE
numpy array: an ndarray of a numeric dtype holds no Python references, so it is untracked. `bytes`
is untracked for the same reason, and on this system that matters twice over, because the ONNX
serialisation buffer IS a `bytes`. Measured on four known objects: both tensors found, both arrays
missing. It also explains, retroactively, why PUMA's earlier gc census accounted 437.9 MB against a
5.63 GB object.

**2. Untracked is not unreachable.** An untracked leaf still hangs off something the collector does
track, so walking `gc.get_referents` TRANSITIVELY from the tracked set reaches it. Measured: a
64 MiB array, a 32 MiB array inside a dict, an 8 MiB array nested two levels below that, a 96 MiB
`bytes` and a 128 MiB tensor, all found, 316,354 objects visited in **982 ms**. One level is not
enough: stopped at depth 1 the same walk misses the array in the dict.

**3. That is cheap enough to ride along in the published run, and `tracemalloc` was not.** An
earlier version armed `tracemalloc` to get each object's creating LINE, and the run went from about
two minutes to **15:38**, a factor of eight. That cannot sit in a measurement run, because wall
time is what types a step free or paid. It cannot sit in a sibling run either: this campaign
already paid for that lesson once, when an object list taken in a separate run described a moment
the published run never had and the two disagreed by 38.6%.

## Why this needs no external freeze

The rest of the campaign stops the program with SIGSTOP and reads `/proc/<pid>/pagemap` from
outside, so that the object list and the residency describe one instant. Here the census runs
INSIDE the program and gets the same guarantee for free: **it runs in the main thread holding the
GIL, so no other Python thread can allocate while it walks.** It then reads `/proc/self/pagemap`
for the ranges it has just enumerated, so the list and its residency come from one instant by
construction.

What the GIL does not stop is a native thread inside a C extension. That is the boundary of this
instrument and it is stated rather than hidden: this census covers Python-owned memory, the C
recorder covers the rest, and the two are reported side by side.

## Why a signal, and why the handler is the safe place

CPython runs a signal handler in the MAIN THREAD, between two bytecodes, never inside a C call.
That is exactly the safe point an earlier version's background thread did not have. That version
killed the measured program outright (`SystemError: tupleobject.c:964`, party 0 on SIGABRT, job
46002533) and, more quietly, ran program code that would not otherwise have run at that moment:
walking live objects means touching them, and touching an object can run `__getattr__`, a property
or a deprecation shim. An instrument that changes the program measures a different program.

## What a row says

    PYOBJ <resident bytes> <size bytes> <kind> <detail> held_by=<owner>

`held_by` is the object that referenced it, and it is the name a step is aimed at: rule 4 asks
whether one change lifts a whole group, which is a question about the owner rather than about the
type. It costs nothing, because the walk already has the parent in hand when it reaches the child.
"""
import gc
import os
import sys

_ENABLED = os.environ.get("SNNI_PY_CENSUS") == "1"
_OUT_DIR = os.environ.get("SNNI_PY_CENSUS_DIR") or "/tmp"
# THE SAME FLOOR AS THE C RECORDER, 64 KiB, and matching it is the point rather than a detail.
# The campaign chose that floor by a stability argument (halve it until the ranking stops moving)
# and every published table uses it. A census with a higher floor cannot be compared to the
# recorder's table at all: measured on this system's baseline, the recorder reports 2766.5 MB over
# 504 allocations at the frozen peak while a 1 MiB census reported 1717.4 MB over 241 objects, and
# most of that gap was simply the objects the higher floor hid. With the floors equal the two
# numbers answer the same question from two independent directions, which is the reconciliation
# this procedure asks of every decomposition.
_MIN = int(os.environ.get("SNNI_PY_CENSUS_MIN", str(1 << 16)))
_PAGE = os.sysconf("SC_PAGE_SIZE")
# Bit 63 of a pagemap entry is `present`, i.e. the high bit of its eighth byte. This maps that byte
# to 1 or 0 so the count is one C call instead of a loop over every page.
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
    """Bytes of [addr, addr+size) the kernel has actually backed, or -1.

    The same question `pmsample` asks from outside, asked from inside so that it lands in the same
    instant as the walk. One 8-byte entry per page; bit 63 is `present`. The first and last pages
    are counted whole, which is the convention the external sampler uses, so the error is bounded
    by two pages per object.
    """
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
    # COUNTED IN C, NOT IN A PYTHON LOOP, and that is the difference between an instrument that
    # rides along and one that does not. A 866 MB object is 211k pages; counting them one at a
    # time in Python made a single census cost seconds and the whole run +49% wall. Slicing every
    # eighth byte, translating it to 0/1 and calling `count` are all C-level, and the same census
    # then costs milliseconds.
    return _PRESENT_TABLE and buf[7::8].translate(_PRESENT_TABLE).count(1) * _PAGE


def _shape(shape):
    """`(28996,768)` and never `(28996, 768)`.

    A row is read field by field, so a space inside one field shifts every field after it. The
    first parse of this file put the holder in the shape's column and read it as an object name.
    """
    return "(" + ",".join(str(int(x)) for x in shape) + ")"


_TENSOR_TYPES = {}


def _is_tensor(t, torch):
    """Is `t` a tensor type, subclasses included, WITHOUT running user code per object.

    The rule everywhere else in this file is exact-type comparison, because `isinstance` consults a
    metaclass hook and a hook is user code. But exact typing missed the one thing that matters most
    here: `torch.nn.Parameter` is a SUBCLASS of `torch.Tensor`, and a model's weights are
    Parameters. Measured, and this is how the omission surfaced: the census reported 980 MB of
    allocations that torch had made and no Python object referred to, and their sizes decoded to
    48 x (768,3072), 98 x (768,768) and 2 x (28996,768) in float32, i.e. exactly the plaintext BERT
    weights, which the baseline plainly does hold.

    So the check is made ONCE PER TYPE and cached. `issubclass` runs on the type object, not on the
    instance, and it runs a few dozen times over a whole census instead of half a million.
    """
    k = _TENSOR_TYPES.get(t)
    if k is None:
        try:
            k = issubclass(t, torch.Tensor)
        except Exception:
            k = False
        _TENSOR_TYPES[t] = k
    return k


def _describe(parent):
    """A short, safe name for the object that held a leaf.

    Type name only. Nothing here calls user code: `type()` cannot run `__getattr__` or a property,
    which is the rule this whole module is built around.
    """
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

    # THE ROOTS ARE SWEPT LAST, and the order is the whole point. A tensor is tracked, so it turns
    # up in `gc.get_objects()` itself; recording it there marks it seen and the walk never reaches
    # it through the object that HOLDS it, so every such row reads `held_by=NoneType`. Measured on
    # this system's baseline: all 254 rows, including the 178.2 MB parameter share, had no holder.
    # Walking the referents first gives each leaf the parent it was reached through, and the sweep
    # afterwards only picks up what nothing tracked points at.
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

    # ONE BUFFER, ONE ROW. Two Python objects can be two views of the SAME bytes, and on this
    # system that is not a corner case: the spill lever does `torch.from_numpy(arr)`, so the array
    # and the tensor share a buffer and each would report its full size. Counted twice, the census
    # totalled 2.34 GB of resident objects inside a 1.72 GB process, which is impossible and is how
    # the defect announced itself. The rows are kept, because both objects are real and a reader
    # wants to see the aliasing; only the SECOND one stops contributing to the total.
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

    # WHAT THE RECORDER SEES AND THIS CENSUS DOES NOT, measured in the same instant and in the
    # same process rather than argued about afterwards.
    #
    # The C recorder keeps a live table of `(pointer, size, site)` for every allocation above the
    # floor, in a file-backed mapping this process owns. Reading it HERE is the only way the two
    # instruments can be compared at ONE moment: the table's FILE holds whatever was live when the
    # process EXITED, and joining that against a census taken at the peak compares two different
    # programs. Measured, and it is why this exists: the exit-time table held 360 live allocations
    # at 656.7 MB while the recorder's own peak snapshot reported 2777.1 MB.
    #
    # An allocation counts as NAMED when a Python object of this census lies inside it. What is
    # left is memory torch allocated that no Python object refers to, and on this system that is
    # the quantity worth knowing: it bounds what a Python-level census can ever explain.
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
        # THE INSTRUMENT CHECKS ITSELF. Resident object bytes cannot exceed the process's own
        # resident size; if they do, something is counted twice and the table is wrong in a way
        # that reads perfectly plausible. It is announced in the block rather than left to a
        # reader's arithmetic.
        over = "" if rss <= 0 or total_res <= rss * 1024 else "|OVER_RSS"
        print(f"PYCENSUS|phase={phase}|rss_kb={rss}|objects={len(rows)}"
              f"|visited={visited}|size_bytes={sum(r[0] for r in rows)}"
              f"|resident_bytes={total_res}"
              f"|covered={0 if rss <= 0 else round(100.0 * total_res / (rss * 1024), 1)}%{over}"
              f"|rec_matched={matched_n}/{matched_b}|rec_unmatched={unmatched_n}/{unmatched_b}",
              file=fh, flush=True)
        # THE ADDRESS IS PRINTED because it is what lets this table be JOINED to the C recorder's,
        # and that join is the coverage check. Measured on the baseline before it existed: the
        # recorder reports 2777.1 MB of live allocations at the frozen peak and this census reaches
        # 1361.4 MB of them, so about 1.4 GB of storage is held by something that is not a Python
        # object. Without addresses that gap is a number; with them it is a list.
        # THE SHAPE OF WHAT PYTHON CANNOT SEE. A count and a byte total say how much is missing;
        # the size histogram says WHAT it is, because a protocol's buffers come in a few exact
        # sizes and a fixed count. Measured here: the unmatched set holds steady at ~267
        # allocations and ~980 MB across every block of a run, which is a standing pool rather
        # than churn.
        for sz, n in sorted(unmatched_sz.items(), key=lambda kv: -kv[0] * kv[1])[:12]:
            print(f"RECGAP {sz * n} {n} {sz}", file=fh, flush=True)
        for size, kind, detail, held, _obj, res, alias, addr in rows:
            print(f"PYOBJ {res} {size} {kind} {detail} held_by={held} {alias} 0x{addr:x}",
                  file=fh, flush=True)


_BUSY = [False]
_LAST = [0.0, 0]          # monotonic seconds, rss_kb at the last census that ran
# The census decides for itself when a signal is worth acting on, and the poller signals freely.
# The policy belongs here rather than in the shared poller because only this side knows what a
# census costs: about a second on this program, against a poller that samples every 15 ms.
_MIN_GAP_S = float(os.environ.get("SNNI_PY_CENSUS_MIN_GAP_S", "3"))
_MIN_GROW = float(os.environ.get("SNNI_PY_CENSUS_MIN_GROW", "1.08"))


def _on_signal(_sig, _frame):
    """Answer a poller signal, or decline it, and never re-enter.

    WHY THE GUARD EXISTS, and it cost a run to find. CPython runs a signal handler between two
    bytecodes INCLUDING while another handler is running, so a second signal arriving during a
    census starts a second census on top of the first. Measured: the poller sent 175 signals over
    eight minutes, every one of them restarted the walk, not a single census ever finished, no
    block was ever written, and the program did not get past its own start line.

    So: one census at a time, and only when the process has actually grown since the last one.
    """
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
    """Answer SIGUSR1 with a census. Must be called from the main thread.

    AND MAKE THE SIGNAL HARMLESS AT SHUTDOWN, which cost a run to learn. The default action for
    SIGUSR1 is to KILL the process. While the handler is installed that does not matter, but during
    interpreter teardown the Python-level callback can no longer run, and a watcher that sends one
    more signal in that window kills a program that had already finished its work: measured, job
    46009906, `RuntimeError: Process process 1 exited with code -10` after the gate had been
    written. So the last thing the process does is set the signal to IGNORE.
    """
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
