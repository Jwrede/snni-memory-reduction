"""Measurement-neutral per-operator markers for CrypTen/SHAFT Graph execution, GPU variant. The
canonical CPU graphop_markers.py plus a per-node GPU pool line (MEM| and GPU|), so the VRAM peak is
attributable to a concrete op. GPU line only when SHAFT_DEVICE=cuda; the counter reads are host-side
(no device sync). Parameter/Constant nodes skipped; layer index parsed from the ONNX node path."""
import os
import re
import sys
import time

import torch
import crypten.nn.module as _ctm

_LAYER_RE = re.compile(r'layers?[._/](\d+)\b')
_SKIP_CLASSES = (_ctm.Parameter, _ctm.Constant)

_USE_CUDA = os.environ.get('SHAFT_DEVICE', 'cuda') == 'cuda' and torch.cuda.is_available()


def _rss_hwm_kb():
    """Current VmRSS and the kernel's exact VmHWM high-water mark, from one read.
    VmHWM is monotone, so emitting it per op node localises this system's transient host peak."""
    rss = hwm = -1
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss = int(line.split()[1])
                elif line.startswith('VmHWM:'):
                    hwm = int(line.split()[1])
                if rss >= 0 and hwm >= 0:
                    break
    except Exception:
        return -1, -1
    return rss, hwm


def _gpu_kb():
    if _USE_CUDA:
        return (int(torch.cuda.memory_allocated() / 1024),
                int(torch.cuda.memory_reserved() / 1024))
    return -1, -1


def _emit(name, cls):
    rank = os.environ.get('RANK', '-1')
    m = _LAYER_RE.search(name)
    layer = m.group(1) if m else '-1'
    clean = name.strip('/').replace('/', '.')
    if clean.endswith('_output_0'):
        clean = clean[: -len('_output_0')]
    ts = int(time.time() * 1000)
    rss, hwm = _rss_hwm_kb()
    print(
        f'MEM|{rank}|{layer}|graphop|{clean}.{cls}|{rss}|{ts}|{hwm}',
        file=sys.stderr, flush=True,
    )
    if _USE_CUDA:
        alloc, reserved = _gpu_kb()
        print(
            f'GPU|{rank}|{layer}|graphop|{clean}.{cls}|{alloc}|{reserved}|{ts}',
            file=sys.stderr, flush=True,
        )


def _wrap_module(mod, name):
    # crypten Module.__getattribute__ re-resolves 'forward' at call time, so capturing `mod.forward`
    # would recurse into our own shadow. Grab the real bound method directly.
    orig = object.__getattribute__(mod, 'forward')
    cls = type(mod).__name__

    def forward(*args, **kwargs):
        out = orig(*args, **kwargs)
        _emit(name, cls)
        return out

    # A plain function attribute shadows the class method for this instance only (crypten
    # Module.__setattr__ routes non-Module values to object.__setattr__).
    mod.forward = forward
    mod._graphop_wrapped = True


_orig_graph_forward = _ctm.Graph.forward


def _instrumented_forward(self, *args, **kwargs):
    if not getattr(self, '_graphop_instrumented', False):
        for name, mod in self._modules.items():
            if isinstance(mod, _SKIP_CLASSES):
                continue
            if getattr(mod, '_graphop_wrapped', False):
                continue
            _wrap_module(mod, name)
        self._graphop_instrumented = True
    return _orig_graph_forward(self, *args, **kwargs)


_ctm.Graph.forward = _instrumented_forward
print('graphop_markers_gpu: Graph.forward instrumented '
      f'(cuda={_USE_CUDA})', file=sys.stderr, flush=True)
