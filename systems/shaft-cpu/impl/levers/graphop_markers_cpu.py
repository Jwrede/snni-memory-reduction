"""Per-operator markers for CrypTen Graph execution, CPU, with VmHWM (instrument).
From shaft-gpu/impl/graphop_markers_cpu.py without the GPU line. Reason: s8_encrypt_chunk moved the
peak into an encoder layer's forward (~545 MB transient per layer; 71.5% anon:residual). VmHWM is
monotone: the operator after which it jumps is the cause. One /proc/self/status read and one stderr
line per node; probe steps only (p_ prefix); cheaper than the pagemap recorder (0.8-4.7% wall,
0.07-0.48% peak).
Emits MEM|<rank>|<layer>|graphop|<node_name>.<OpClass>|<VmRSS_kB>|<ts_ms>|<VmHWM_kB>; Parameter/Constant
nodes skipped.
"""
import os
import re
import sys
import time

import crypten.nn.module as _ctm

_LAYER_RE = re.compile(r'layers?[._/](\d+)\b')
_SKIP_CLASSES = (_ctm.Parameter, _ctm.Constant)

_USE_CUDA = False


def _rss_hwm_kb():
    """Current RSS and the kernel's exact high-water mark, from one read of one file. VmHWM is
    monotone, so per-op emission localises the transient host peak between two named operators; the
    poller reads this peak 18 to 34% low."""
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


def _wrap_module(mod, name):
    # Module.__getattribute__ intercepts 'forward' and re-resolves it at call time, so capturing
    # mod.forward would recurse into our shadow; grab the real bound method directly.
    orig = object.__getattribute__(mod, 'forward')
    cls = type(mod).__name__

    def forward(*args, **kwargs):
        out = orig(*args, **kwargs)
        _emit(name, cls)
        return out

    # Plain function attribute: Module.__setattr__ routes non-Module values to object.__setattr__,
    # shadowing the class method for this instance only.
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
print('LEVER|graphop_markers_cpu|patched', file=sys.stderr, flush=True)
