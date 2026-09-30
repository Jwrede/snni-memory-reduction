"""Print CrypTen's own per-operator-class communication buckets (COMMCLASS|<rank>|<bucket>|<bytes>|
<rounds>) after every graph. PROBE ONLY, loaded via SHAFT_LEVERS in p_ steps, never in a published
row. Reads the buckets CrypTen already maintains (no wrapping), to localise which operator class
holds the 524,288-byte/one-round transcript gap between s5_per_layer and s4_param_defer."""
import os
import sys

import crypten.nn.module as _ctm

_BUCKETS = ("embedding", "matmul", "softmax", "gelu", "layernorm", "tanh", "conv", "other", "total")

_orig_graph_forward = _ctm.Graph.forward


def _instrumented_forward(self, *args, **kwargs):
    out = _orig_graph_forward(self, *args, **kwargs)
    rank = os.environ.get("RANK", "-1")
    for b in _BUCKETS:
        by = getattr(self, f"{b}_comm_bytes", None)
        rd = getattr(self, f"{b}_comm_rounds", None)
        if by is None and rd is None:
            continue
        print(f"COMMCLASS|{rank}|{b}|{by if by is not None else -1}|"
              f"{rd if rd is not None else -1}", file=sys.stderr, flush=True)
    return out


_ctm.Graph.forward = _instrumented_forward

print('graphop_comm_gpu: patched Graph.forward (per-class comm buckets)',
      file=sys.stderr, flush=True)
