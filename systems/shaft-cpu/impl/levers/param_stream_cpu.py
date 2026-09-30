"""Encrypted parameter set streamed from disk instead of held from encryption onward.
Object (probe_graph_owners_cpu.py): 866494480 B over 201 Parameter modules (int64 shares), 32.2% of s3's
2693096 kB peak, largest by 1.6x; all exist before the first operation: reloadable (third condition).
At the first Graph.forward each share is written to a file and dropped; Parameter.forward reads it back.
Gate unchanged (same order, sizes, int64 bytes; no generator; encoder scale carried explicitly).
Real filesystem only (/proc/mounts; tmpfs, ramfs, devtmpfs refused; logged); not mmap'd.
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

_ORIG_GRAPH_FORWARD = cnnmod.Graph.forward
_ORIG_PARAM_FORWARD = cnnmod.Parameter.forward

_SPILLED = {}          # id(module) -> (path, shape, scale)
_DIR = None
_STATS = {"bytes": 0, "n": 0, "loaded": 0}
_BAD_FS = ("tmpfs", "ramfs", "devtmpfs")


def _fstype(path):
    """Filesystem type of the mount that holds `path`, from /proc/mounts."""
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


def _inner_share(val):
    """The ArithmeticSharedTensor behind a parameter value, or None. After encrypt() a parameter is an
    MPCTensor forwarding .share/.encoder to ._tensor (mpc/mpc.py:197,207), so an isinstance test on the
    value matches nothing (the first version did that and silently spilled zero; hence reporting bytes)."""
    if isinstance(val, ArithmeticSharedTensor):
        return val
    t = getattr(val, "_tensor", None)
    return t if isinstance(t, ArithmeticSharedTensor) else None


def _rewrap(template, new_inner):
    """A fresh wrapper of the same type as template around new_inner, drawing nothing. NOT
    shallow_copy(), which builds MPCTensor([]) and could take a PRZS on an empty size; __new__ plus a
    dict copy provably touches no generator, and the gate depends on draw order."""
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


def _spill_all(graph):
    global _DIR
    _DIR = os.path.join(os.environ.get("SNNI_SPILL_DIR", "/tmp"),
                        f"snni_param_spill_{os.getpid()}")
    fs = _fstype(os.path.dirname(_DIR.rstrip("/")) or "/")
    if fs in _BAD_FS:
        print(f"FATAL|param_stream_cpu|spill dir {_DIR} is on {fs}; that is MEMORY, not disk, so "
              f"spilling there would report a reduction that did not happen",
              file=sys.stderr, flush=True)
        raise RuntimeError(f"refusing to spill onto {fs}")
    os.makedirs(_DIR, exist_ok=True)
    # Each party writes ~866 MB to a shared node-local fs; removed on exit whatever the exit was.
    atexit.register(lambda: shutil.rmtree(_DIR, ignore_errors=True))

    for name, mod in graph._modules.items():
        if not isinstance(mod, cnnmod.Parameter):
            continue
        val = getattr(mod, "data", None)
        inner = _inner_share(val)
        if inner is None:
            continue                                   # not encrypted; leave it alone
        t = inner.share
        if t.dtype is not torch.int64 or t.numel() == 0:
            continue
        path = os.path.join(_DIR, f"{_STATS['n']:04d}_{t.numel()}.bin")
        arr = t.detach().numpy()                       # no copy: shares are contiguous int64
        arr.tofile(path)
        _SPILLED[id(mod)] = (path, tuple(t.shape), inner.encoder.scale, val)
        _STATS["bytes"] += t.numel() * t.element_size()
        _STATS["n"] += 1
        # Drop the payload but keep the objects: set_parameter stores the value in both
        # _parameters["data"] and as an instance attribute (nn/module.py:143-144), both the same
        # wrapper, so emptying its inner share frees once. A zero-element share makes a stray read
        # fail loudly instead of returning zeros.
        inner.share = torch.empty(0, dtype=torch.int64)

    print(f"LEVER|param_stream_cpu|spilled_bytes={_STATS['bytes']}|params={_STATS['n']}|"
          f"dir={_DIR}|fstype={_fstype(_DIR)}", file=sys.stderr, flush=True)


_DONE = False


def graph_forward(self, *args):
    global _DONE
    if not _DONE:
        _DONE = True                                   # a flag, not `if not _SPILLED`: a graph with
        _spill_all(self)                               # nothing to spill must not retry every pass
    return _ORIG_GRAPH_FORWARD(self, *args)


def param_forward(self, input):
    rec = _SPILLED.get(id(self))
    if rec is None:
        return _ORIG_PARAM_FORWARD(self, input)
    path, shape, scale, template = rec
    arr = np.fromfile(path, dtype=np.int64)
    t = torch.from_numpy(arr).view(shape)
    inner = ArithmeticSharedTensor.from_shares(t)
    # from_shares() builds a default encoder, so restore the recorded scale explicitly: a wrong scale
    # would be a silent factor-of-65536, not a crash.
    inner.encoder._scale = scale
    _STATS["loaded"] += 1
    return _rewrap(template, inner)


cnnmod.Graph.forward = graph_forward
cnnmod.Parameter.forward = param_forward

print("LEVER|param_stream_cpu|patched", file=sys.stderr, flush=True)
