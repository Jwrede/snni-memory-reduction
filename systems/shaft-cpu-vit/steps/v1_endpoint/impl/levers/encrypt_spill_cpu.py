"""Spill each encrypted parameter the moment it is created, so the 201 shares never coexist.

THE OBJECT, from the owner-grouped ranking at s1's peak (job 45514252): **866494480 B across the
graph's 201 `Parameter` modules**, 28.3% of that peak and the largest object on the system, against
534454272 B for the next one. The per-shape census hid it behind a mixed 712 MB row; grouping by
owner is what makes it visible. `probe_graph_owners_cpu.py` reaches the same 866494480 B from the
graph itself.

WHY THIS IS NOT `param_stream_cpu`, WHICH ATTACKED THE SAME OBJECT AND MEASURED NOTHING. That lever
spills at the first `Graph.forward`, i.e. after `Module.encrypt()` has already created all 201
shares. All 866 MB were therefore touched, and freeing them handed the pages to glibc's arena rather
than to the kernel: measured -0.74% against a 0.812% spread while `anon:[heap]` rose 383.6 -> 555.3
MB. 866 MB released, 20 MB of RSS. Its variants `s4_param_release` (+41% runtime) and the published
`s4_param_mmap_release` (-0.44%) differ only in how hard they push the pages back out.

This one never creates the coexistence. `Module.encrypt()` recurses through
`self._apply(lambda m: m.encrypt(...))` (`nn/module.py:508`), so each module encrypts its own
parameters and only then does the walk move on. Spilling inside that walk means at most ONE share is
resident at a time, and pages that were never touched need no allocator to give them back. That is
PROCEDURE.md's fourth condition, NOT YET NEEDED, rather than its third.

WHY THE DRAWS ARE UNTOUCHED, which is what rules out most levers on this system and is measured
rather than assumed (`GATE-TOLERANCE.md`: with different masks this observable moves by 2.34 on a
value of 0.135 at 12 layers and by 0.30 on 0.148 at one, so a moved draw is not a subtle effect).
Every `crypten.cryptensor()` call happens in the same order, at the same sizes, in the same walk;
this lever adds no call that consults a generator. Writing bytes to a file and reading them back
draws nothing. Encryption still completes entirely before the first inference operation, so no
PRZS draw is interleaved with a Beaver draw.

THE SPILL MUST GO TO A REAL FILESYSTEM. `/dev/shm` and any tmpfs would turn 866 MB of anonymous
memory into 866 MB of shared memory and report a reduction that never happened. The filesystem type
is read from `/proc/mounts`, logged on every run, and the lever REFUSES tmpfs, ramfs and devtmpfs.
The file is not mmap'd either: a mapped file is counted by `smaps_objects.py` as `file:`, which is
excluded from `W` as library overhead, so mmap would move the bytes out of the number being reduced
instead of out of memory.
"""

import atexit
import os
import shutil
import sys

import numpy as np
import torch

import crypten
from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
from crypten.nn import module as cnnmod

_ORIG_ENCRYPT = cnnmod.Module.encrypt
_ORIG_PARAM_FORWARD = cnnmod.Parameter.forward

_SPILLED = {}          # id(module) -> (path, shape, scale, template)
_DIR = None
_STATS = {"bytes": 0, "n": 0, "loaded": 0, "peak_resident": 0}
_BAD_FS = ("tmpfs", "ramfs", "devtmpfs")


def _fstype(path):
    """Filesystem type of the mount holding `path`, from /proc/mounts."""
    best, best_len = "unknown", -1
    real = os.path.realpath(path)
    try:
        for ln in open("/proc/mounts"):
            f = ln.split()
            if len(f) < 3:
                continue
            mp = f[1]
            if (real == mp or real.startswith(mp.rstrip("/") + "/")) and len(mp) > best_len:
                best, best_len = f[2], len(mp)
    except OSError:
        pass
    return best


def _ensure_dir():
    global _DIR
    if _DIR is not None:
        return
    base = os.environ.get("SNNI_SPILL_DIR", "/tmp")
    fs = _fstype(base)
    if fs in _BAD_FS:
        print(f"FATAL|encrypt_spill_cpu|spill base {base} is on {fs}; that is MEMORY, not disk, so "
              f"spilling there would report a reduction that did not happen",
              file=sys.stderr, flush=True)
        raise RuntimeError(f"refusing to spill onto {fs}")
    _DIR = os.path.join(base, f"snni_encrypt_spill_{os.getpid()}")
    os.makedirs(_DIR, exist_ok=True)
    # Node-local scratch shared with every other job on the node, ~866 MB per party. Removed on any
    # exit rather than left behind.
    atexit.register(lambda: shutil.rmtree(_DIR, ignore_errors=True))


def _inner_share(val):
    """The ArithmeticSharedTensor behind a parameter value, or None.

    After encryption a parameter is an `MPCTensor`, which forwards `.share` and `.encoder` to
    `._tensor` (`mpc/mpc.py:197,207`). Testing for `ArithmeticSharedTensor` alone finds nothing and
    yields a lever that spills zero bytes while the run completes and the gate passes -- which is
    why the byte count is announced rather than only the fact of patching.
    """
    if isinstance(val, ArithmeticSharedTensor):
        return val
    t = getattr(val, "_tensor", None)
    return t if isinstance(t, ArithmeticSharedTensor) else None


def _rewrap(template, new_inner):
    """A fresh wrapper of the template's type around `new_inner`, drawing nothing.

    NOT `shallow_copy()`: that constructs `MPCTensor([])`, which runs the constructor and would take
    a PRZS on an empty size. A zero-element draw almost certainly does not advance the generators,
    and "almost certainly" is not good enough where the gate depends on the draw order.
    """
    if isinstance(template, ArithmeticSharedTensor):
        return new_inner
    cls = type(template)
    new = cls.__new__(cls)
    try:
        new.__dict__.update(template.__dict__)
    except AttributeError:
        return template.shallow_copy()
    new._tensor = new_inner
    return new


def _spill_one(mod):
    """Write this parameter's share out and drop the payload. At most one is resident."""
    val = getattr(mod, "data", None)
    inner = _inner_share(val)
    if inner is None:
        return
    t = inner.share
    if t.dtype is not torch.int64 or t.numel() == 0:
        return
    _ensure_dir()
    path = os.path.join(_DIR, f"{_STATS['n']:04d}_{t.numel()}.bin")
    t.detach().numpy().tofile(path)                    # shares are contiguous int64, no copy
    _SPILLED[id(mod)] = (path, tuple(t.shape), inner.encoder.scale, val)
    _STATS["bytes"] += t.numel() * t.element_size()
    _STATS["n"] += 1
    _STATS["peak_resident"] = max(_STATS["peak_resident"], t.numel() * t.element_size())
    # `set_parameter` stores the value in `_parameters["data"]` AND as an instance attribute
    # (`nn/module.py:143-144`), both pointing at one wrapper, so emptying the wrapper's inner share
    # releases the memory once and leaves nothing dangling. A zero-element share makes a stray READ
    # fail loudly rather than return zeros, which would be a wrong answer that looked right.
    inner.share = torch.empty(0, dtype=torch.int64)


def encrypt(self, mode=True, src=0):
    out = _ORIG_ENCRYPT(self, mode=mode, src=src)
    # AFTER this module's own parameters are encrypted and BEFORE the walk reaches the next one.
    # Only `mode=True`: decryption is the inverse and must not be intercepted.
    if mode and isinstance(self, cnnmod.Parameter):
        _spill_one(self)
    return out


def param_forward(self, input):
    rec = _SPILLED.get(id(self))
    if rec is None:
        return _ORIG_PARAM_FORWARD(self, input)
    path, shape, scale, template = rec
    arr = np.fromfile(path, dtype=np.int64)
    t = torch.from_numpy(arr).view(shape)
    inner = ArithmeticSharedTensor.from_shares(t)
    # `from_shares()` builds a default encoder. The recorded scale is put back explicitly rather
    # than assumed to match: a wrong scale is a silent factor of 65536, not a crash.
    inner.encoder._scale = scale
    _STATS["loaded"] += 1
    return _rewrap(template, inner)


def _report():
    print(f"LEVER|encrypt_spill_cpu|spilled_bytes={_STATS['bytes']}|params={_STATS['n']}|"
          f"max_one_share={_STATS['peak_resident']}|loaded={_STATS['loaded']}|"
          f"dir={_DIR}|fstype={_fstype(_DIR) if _DIR else 'none'}", file=sys.stderr, flush=True)


atexit.register(_report)

cnnmod.Module.encrypt = encrypt
cnnmod.Parameter.forward = param_forward

print("LEVER|encrypt_spill_cpu|patched", file=sys.stderr, flush=True)
