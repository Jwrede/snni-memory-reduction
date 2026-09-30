"""Each encrypted parameter spilled when created (the 201 shares never coexist).
Object (owner-grouped ranking at s1's peak, job 45514252): 866494480 B over the 201 Parameter modules,
28.3%, largest object (the per-shape census hid it in a mixed 712 MB row). Not param_stream_cpu (spilled
after all 201 existed; arena reuse; -0.74% vs 0.812% spread). Spill inside Module.encrypt()'s _apply
walk (nn/module.py:508): at most one share resident. Draws unchanged (same cryptensor calls, order,
sizes; encryption completes before the first inference; GATE-TOLERANCE.md: different masks move the
observable 2.34 on 0.135 at 12 layers). Spill to a real filesystem only (/proc/mounts; tmpfs, ramfs,
devtmpfs refused; logged). Not mmap'd (file: is excluded from W).
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
    # Node-local scratch, ~866 MB per party, removed on any exit.
    atexit.register(lambda: shutil.rmtree(_DIR, ignore_errors=True))


def _inner_share(val):
    """The ArithmeticSharedTensor behind a parameter value, or None. After encryption a parameter is
    an MPCTensor forwarding .share/.encoder to ._tensor (mpc/mpc.py:197,207), so testing for
    ArithmeticSharedTensor alone would spill zero bytes while the gate passes."""
    if isinstance(val, ArithmeticSharedTensor):
        return val
    t = getattr(val, "_tensor", None)
    return t if isinstance(t, ArithmeticSharedTensor) else None


def _rewrap(template, new_inner):
    """A fresh wrapper of the template's type around `new_inner`, drawing nothing. NOT shallow_copy(),
    which constructs MPCTensor([]) and would take a PRZS on an empty size (a zero-element draw almost
    certainly does not advance the generators, but "almost certainly" is not good enough here)."""
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
    # set_parameter stores the value in _parameters["data"] AND as an instance attribute
    # (nn/module.py:143-144), both one wrapper, so emptying the inner share releases once; a
    # zero-element share makes a stray READ fail loudly rather than return zeros.
    inner.share = torch.empty(0, dtype=torch.int64)


def encrypt(self, mode=True, src=0):
    out = _ORIG_ENCRYPT(self, mode=mode, src=src)
    # After this module's own parameters are encrypted, before the walk reaches the next. Only
    # mode=True: decryption is the inverse and must not be intercepted.
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
    # from_shares builds a default encoder; the recorded scale is put back explicitly (a wrong scale
    # is a silent factor of 65536, not a crash).
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
