"""Spills each encrypted parameter to disk at encryption; reloads it onto the device at use.
Target: 866.6 MB of int64 parameter shares made device-resident by private_model.cuda() (54.7% of
s2_embed_chunk's 1,546,240 kB peak). = encrypt_spill_cpu's disk backing + param_stream_gpu's device
handling (alone rejected: host 2.93 -> 4.10 GB). Patches: Module.encrypt spills after each
Parameter; Parameter.cuda is a no-op; Parameter.forward reads the file and moves it to the device.
Same bytes, once each, same order. Refuses tmpfs."""

import atexit
import ctypes
import os
import shutil
import sys

import numpy as np
import torch

import crypten
import crypten.nn.module as _ctm
from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor

_ENABLED = os.environ.get("SHAFT_PARAM_SPILL", "0") == "1"

_ORIG_ENCRYPT = _ctm.Module.encrypt
_ORIG_PARAM_FORWARD = _ctm.Parameter.forward
_ORIG_PARAM_CUDA = getattr(_ctm.Parameter, "cuda", None)

_SPILLED = {}          # id(module) -> (path, shape, scale, template)
_DIR = None
_STATS = {"bytes": 0, "n": 0, "loaded": 0, "max_one_share": 0}
_BAD_FS = ("tmpfs", "ramfs", "devtmpfs")

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


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
        print(f"FATAL|param_spill_gpu|spill base {base} is on {fs}; that is MEMORY, not disk, so "
              f"spilling there would report a reduction that did not happen",
              file=sys.stderr, flush=True)
        raise RuntimeError(f"refusing to spill onto {fs}")
    _DIR = os.path.join(base, f"snni_param_spill_{os.getpid()}")
    os.makedirs(_DIR, exist_ok=True)
    atexit.register(lambda: shutil.rmtree(_DIR, ignore_errors=True))


def _inner_share(val):
    """The ArithmeticSharedTensor behind a parameter value, or None. After encryption a parameter is
    an MPCTensor forwarding `.share`/`.encoder` to `._tensor`, so testing for ArithmeticSharedTensor
    alone would find nothing and spill zero bytes silently (hence the byte count is announced)."""
    if isinstance(val, ArithmeticSharedTensor):
        return val
    t = getattr(val, "_tensor", None)
    return t if isinstance(t, ArithmeticSharedTensor) else None


def _rewrap(template, new_inner):
    """A fresh wrapper of the template's type around `new_inner`, drawing nothing. NOT shallow_copy():
    that constructs MPCTensor([]), taking a PRZS on an empty size, and a zero-element draw might
    advance the generators, which the gate's draw-order dependence cannot tolerate."""
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
    # This lever runs before `.cuda()`, so a share already on the device means the ordering
    # assumption is wrong.
    if getattr(t, "is_cuda", False):
        print("FATAL|param_spill_gpu|a parameter share is already on the device at encrypt time; "
              "this lever assumes it runs before private_model.cuda()", file=sys.stderr, flush=True)
        raise RuntimeError("param_spill_gpu: share already on device")
    if t.dtype is not torch.int64 or t.numel() == 0:
        return
    _ensure_dir()
    path = os.path.join(_DIR, f"{_STATS['n']:04d}_{t.numel()}.bin")
    t.detach().numpy().tofile(path)                    # shares are contiguous int64, no copy
    _SPILLED[id(mod)] = (path, tuple(t.shape), inner.encoder.scale, val)
    nb = t.numel() * t.element_size()
    _STATS["bytes"] += nb
    _STATS["n"] += 1
    _STATS["max_one_share"] = max(_STATS["max_one_share"], nb)
    # A zero-element share makes a stray read fail loudly rather than return zeros.
    inner.share = torch.empty(0, dtype=torch.int64)


def encrypt(self, mode=True, src=0):
    out = _ORIG_ENCRYPT(self, mode=mode, src=src)
    # Only mode=True: decryption is the inverse and must not be intercepted.
    if mode and isinstance(self, _ctm.Parameter):
        _spill_one(self)
    return out


def parameter_cuda(self, device=None):
    """A Parameter ignores `.cuda()`: there is nothing on the host to move."""
    return self


def param_forward(self, input):
    rec = _SPILLED.get(id(self))
    if rec is None:
        return _ORIG_PARAM_FORWARD(self, input)
    path, shape, scale, template = rec
    arr = np.fromfile(path, dtype=np.int64)
    t = torch.from_numpy(arr).view(shape)
    inner = ArithmeticSharedTensor.from_shares(t)
    # from_shares() builds a default encoder; the recorded scale is put back explicitly (a wrong
    # scale is a silent factor of 65536, not a crash).
    inner.encoder._scale = scale
    out = _rewrap(template, inner)
    # The device copy is what the graph gets; the host tensor dies with this frame. .cuda() is
    # in-place, but this object was built one line ago and owned by nobody else.
    out.cuda()
    _STATS["loaded"] += 1
    return out


def _report():
    print(f"LEVER|param_spill_gpu|spilled_bytes={_STATS['bytes']}|params={_STATS['n']}|"
          f"max_one_share={_STATS['max_one_share']}|loaded={_STATS['loaded']}|"
          f"dir={_DIR}|fstype={_fstype(_DIR) if _DIR else 'none'}", file=sys.stderr, flush=True)


if _ENABLED:
    atexit.register(_report)
    _ctm.Module.encrypt = encrypt
    _ctm.Parameter.cuda = parameter_cuda
    _ctm.Parameter.forward = param_forward
    # ANNOUNCEMENT IN THIS SYSTEM'S FORM: the sbatch's generic slot greps for `<name>: patched`.
    print("param_spill_gpu: patched", file=sys.stderr, flush=True)
else:
    print("param_spill_gpu: inactive (SHAFT_PARAM_SPILL != 1)", file=sys.stderr, flush=True)
