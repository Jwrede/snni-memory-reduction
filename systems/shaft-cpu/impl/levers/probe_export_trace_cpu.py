"""Diagnostic only. Brackets torch.onnx.export (probe_convert_trace max 1612240 kB vs peak 2064884 kB;
452 MB gap in _export_pytorch_model, before _load_onnx_model). RSS and live torch storage before and
after; RSS sampled by a thread during the export. Never in a measured run.
"""

import gc
import os
import sys
import threading
import time

import torch

from crypten.nn import onnx_converter

_OUT = None
_ORIG = onnx_converter._export_pytorch_model
_N = 0


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


def _torch_bytes():
    seen = {}
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor):
                continue
            st = obj.untyped_storage()
            seen[st.data_ptr()] = st.nbytes()
        except Exception:
            continue
    return len(seen), sum(seen.values())


def _mark(tag):
    n, b = _torch_bytes()
    print(f"EXPORT|{tag}|rss_kb={_rss_kb()}|hwm_kb={_hwm_kb()}|torch_tensors={n}|torch_bytes={b}",
          file=_OUT, flush=True)


def _export_pytorch_model(f, pytorch_model, dummy_input, **kwargs):
    global _N
    _N += 1
    tag = f"export{_N}"
    _mark(f"{tag}_before")

    stop = threading.Event()
    peak = [0]

    def sampler():
        # RSS only, no gc walk: this runs while the export is in flight and must not perturb it.
        while not stop.is_set():
            r = _rss_kb()
            if r > peak[0]:
                peak[0] = r
            time.sleep(0.01)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        out = _ORIG(f, pytorch_model, dummy_input, **kwargs)
    finally:
        stop.set()
        t.join(timeout=1.0)
    print(f"EXPORT|{tag}_inflight_max_rss_kb={peak[0]}", file=_OUT, flush=True)
    _mark(f"{tag}_after")
    return out


def _start():
    global _OUT
    pid = os.getpid()
    d = os.environ.get("RESULTS_DIR", "/tmp")
    _OUT = open(os.path.join(d, f"exporttrace_pid{pid}.txt"), "w", buffering=1)
    print(f"LEVER|probe_export_trace_cpu|patched pid={pid}", file=sys.stderr, flush=True)


_start()
onnx_converter._export_pytorch_model = _export_pytorch_model
