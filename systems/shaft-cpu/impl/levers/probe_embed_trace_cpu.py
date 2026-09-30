"""Diagnostic only. Where inside the embedding matmul RSS peaks. probe_census_cpu's last census = the
highest sampled maximum (polled maximum 5% below VmHWM). Wraps only the embedding's Beaver matmul
(identified by operand shape) and records RSS at every allocation boundary; gc census only here (nine
walks). Never in a measured run.
"""

import gc
import os
import sys

import torch

import crypten
from crypten.mpc.primitives import beaver
from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor

_ORIG_MATMUL = beaver.matmul
_OUT = None
_BIG = None  # the operand shape that identifies the embedding matmul, learned at first sight


def _rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def _hwm_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return -1


def _live(shape, dtype=torch.int64):
    """(count, bytes) of live tensors with this exact shape and dtype, deduplicated by storage."""
    seen = {}
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor):
                continue
            if obj.dtype is not dtype or tuple(obj.shape) != tuple(shape):
                continue
            st = obj.untyped_storage()
            seen[st.data_ptr()] = st.nbytes()
        except Exception:
            continue
    return len(seen), sum(seen.values())


def _mark(tag, shape):
    n, b = _live(shape)
    print(f"EMBED|{tag}|rss_kb={_rss_kb()}|hwm_kb={_hwm_kb()}|"
          f"live_{tuple(shape)}_n={n}|bytes={b}", file=_OUT, flush=True)


def matmul(x, y, *args, **kwargs):
    # The embedding is the only matmul whose right operand is the whole vocabulary table (everything
    # else in BERT-base is at most (3072,768)); selecting by shape survives a lever changing matmul order.
    ysz = tuple(y.size())
    if len(ysz) != 2 or ysz[0] < 10000:
        return _ORIG_MATMUL(x, y, *args, **kwargs)

    shape = ysz
    _mark("entry", shape)
    provider = crypten.mpc.get_default_provider()
    a, b, c = provider.generate_additive_triple(
        x.size(), y.size(), "matmul", device=x.device, *args, **kwargs
    )
    _mark("after_triple", shape)

    from crypten.mpc.primitives.beaver import IgnoreEncodings
    with IgnoreEncodings([a, b, x, y]):
        d0 = x - a
        _mark("after_x_minus_a", shape)
        d1 = y - b
        _mark("after_y_minus_b", shape)
        epsilon, delta = ArithmeticSharedTensor.reveal_batch([d0, d1])
    _mark("after_reveal", shape)
    del d0, d1

    c._tensor += torch.matmul(epsilon, b._tensor, *args, **kwargs)
    _mark("after_eps_b", shape)
    c._tensor += torch.matmul(a._tensor, delta, *args, **kwargs)
    _mark("after_a_delta", shape)
    c += torch.matmul(epsilon, delta, *args, **kwargs)
    _mark("after_eps_delta", shape)
    return c


def _start():
    global _OUT
    pid = os.getpid()
    d = os.environ.get("RESULTS_DIR", "/tmp")
    _OUT = open(os.path.join(d, f"embedtrace_pid{pid}.txt"), "w", buffering=1)
    print(f"LEVER|probe_embed_trace_cpu|patched pid={pid}", file=sys.stderr, flush=True)


_start()
beaver.matmul = matmul
