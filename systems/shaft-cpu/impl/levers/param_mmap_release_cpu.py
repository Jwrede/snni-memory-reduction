"""Encrypted parameter set spilled and its pages released, without slowing inference.
Supersedes param_stream_cpu.py (byte-identical for its lever_md5); use one. Third attempt on 866494480 B:
spill alone left RSS unchanged (2.36 MB shares in glibc's arena); global MALLOC_MMAP_THRESHOLD_ fixed it
at +35% wall. Here mallopt only for the outermost encrypt() (131072 on enter, 33554432 on leave =
glibc's dynamic ceiling), malloc_trim(0) after the spill. Gate unchanged. Both parts reported.
"""

import atexit
import ctypes
import os
import shutil
import sys

import numpy as np
import torch

from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
from crypten.nn import module as cnnmod

_ORIG_GRAPH_FORWARD = cnnmod.Graph.forward
_ORIG_PARAM_FORWARD = cnnmod.Parameter.forward

_SPILLED = {}
_DIR = None
_STATS = {"bytes": 0, "n": 0, "loaded": 0}
_BAD_FS = ("tmpfs", "ramfs", "devtmpfs")
_DONE = False
_ENCRYPT_DEPTH = 0

_M_MMAP_THRESHOLD = -3          # glibc malloc.h
_SPILL_THRESHOLD = 131072       # while encrypting: mmap every share so free() munmaps it
_RESTORE_THRESHOLD = 33554432   # afterwards: where glibc's own dynamic threshold tops out
_LIBC = None
try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError as _exc:         # pragma: no cover
    print(f"LEVER|param_mmap_release_cpu|libc unavailable: {_exc}", file=sys.stderr, flush=True)


def _mallopt(value, tag):
    if _LIBC is None:
        return
    rc = _LIBC.mallopt(_M_MMAP_THRESHOLD, ctypes.c_int(value))
    print(f"LEVER|param_mmap_release_cpu|mallopt M_MMAP_THRESHOLD={value} rc={rc} ({tag})",
          file=sys.stderr, flush=True)


_ORIG_ENCRYPT = cnnmod.Module.encrypt


def encrypt(self, mode=True, src=0):
    """Hold the mmap threshold low for the whole of the outermost encrypt only; a depth counter makes
    the window exactly the top-level call (Module.encrypt recurses through _apply)."""
    global _ENCRYPT_DEPTH
    outer = _ENCRYPT_DEPTH == 0 and mode
    if outer:
        _mallopt(_SPILL_THRESHOLD, "entering encrypt")
    _ENCRYPT_DEPTH += 1
    try:
        return _ORIG_ENCRYPT(self, mode=mode, src=src)
    finally:
        _ENCRYPT_DEPTH -= 1
        if outer:
            _mallopt(_RESTORE_THRESHOLD, "leaving encrypt")


def _fstype(path):
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


def _rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def _inner_share(val):
    """The ArithmeticSharedTensor behind a parameter value, or None. After encrypt() a parameter is an
    MPCTensor forwarding .share/.encoder to ._tensor (mpc/mpc.py:197,207), so an isinstance test on
    the value matches nothing (the first version did that and silently spilled zero)."""
    if isinstance(val, ArithmeticSharedTensor):
        return val
    t = getattr(val, "_tensor", None)
    return t if isinstance(t, ArithmeticSharedTensor) else None


def _rewrap(template, new_inner):
    """A fresh wrapper of the same type around new_inner, drawing nothing. NOT shallow_copy(), which
    builds MPCTensor([]) and could take a PRZS on an empty size, and the gate depends on draw order."""
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


def _trim():
    """Return free arena pages to the kernel, and report what it actually achieved."""
    before = _rss_kb()
    try:
        rc = ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError as exc:
        print(f"LEVER|param_mmap_release_cpu|malloc_trim unavailable: {exc}", file=sys.stderr, flush=True)
        return
    after = _rss_kb()
    print(f"LEVER|param_mmap_release_cpu|malloc_trim rc={rc}|rss_kb {before} -> {after}|"
          f"returned_kb={before - after}", file=sys.stderr, flush=True)


def _spill_all(graph):
    global _DIR
    base = os.environ.get("SNNI_SPILL_DIR", "/tmp")
    fs = _fstype(base)
    if fs in _BAD_FS:
        print(f"FATAL|param_mmap_release_cpu|spill dir {base} is on {fs}; that is MEMORY, not disk, so "
              f"spilling there would report a reduction that did not happen",
              file=sys.stderr, flush=True)
        raise RuntimeError(f"refusing to spill onto {fs}")
    _DIR = os.path.join(base, f"snni_param_spill_{os.getpid()}")
    os.makedirs(_DIR, exist_ok=True)
    atexit.register(lambda: shutil.rmtree(_DIR, ignore_errors=True))

    for name, mod in graph._modules.items():
        if not isinstance(mod, cnnmod.Parameter):
            continue
        val = getattr(mod, "data", None)
        inner = _inner_share(val)
        if inner is None:
            continue
        t = inner.share
        if t.dtype is not torch.int64 or t.numel() == 0:
            continue
        path = os.path.join(_DIR, f"{_STATS['n']:04d}_{t.numel()}.bin")
        t.detach().numpy().tofile(path)
        _SPILLED[id(mod)] = (path, tuple(t.shape), inner.encoder.scale, val)
        _STATS["bytes"] += t.numel() * t.element_size()
        _STATS["n"] += 1
        inner.share = torch.empty(0, dtype=torch.int64)

    print(f"LEVER|param_mmap_release_cpu|spilled_bytes={_STATS['bytes']}|params={_STATS['n']}|"
          f"dir={_DIR}|fstype={_fstype(_DIR)}|"
          f"mmap_threshold=mallopt({_SPILL_THRESHOLD} during encrypt)",
          file=sys.stderr, flush=True)
    _trim()


def graph_forward(self, *args):
    global _DONE
    if not _DONE:
        _DONE = True
        _spill_all(self)
    return _ORIG_GRAPH_FORWARD(self, *args)


def param_forward(self, input):
    rec = _SPILLED.get(id(self))
    if rec is None:
        return _ORIG_PARAM_FORWARD(self, input)
    path, shape, scale, template = rec
    t = torch.from_numpy(np.fromfile(path, dtype=np.int64)).view(shape)
    inner = ArithmeticSharedTensor.from_shares(t)
    # Scale restored explicitly: a wrong scale would be a silent factor-of-65536, not a crash.
    inner.encoder._scale = scale
    _STATS["loaded"] += 1
    return _rewrap(template, inner)


cnnmod.Module.encrypt = encrypt
cnnmod.Graph.forward = graph_forward
cnnmod.Parameter.forward = param_forward

print("LEVER|param_mmap_release_cpu|patched", file=sys.stderr, flush=True)
