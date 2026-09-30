"""Large-matmul chunk lever (step s5): split the FFN up-projection MatMul.forward along its output
dim into SHAFT_MATMUL_CHUNKS blocks (empty_cache between, then concat). Targets s2's VRAM wall, the
~615 MiB Beaver transient inside hidden[128,768] @ W[768,3072]. Only output dim >= SHAFT_MATMUL_MIN_N
(default 3072) is chunked. Block matmul is exact, so memory changes, not the result. Import after crypten."""
import os
import sys
import gc

import torch
import crypten
import crypten.nn.module as _ctmod

_NCH = int(os.environ.get("SHAFT_MATMUL_CHUNKS", "1"))
_MIN_N = int(os.environ.get("SHAFT_MATMUL_MIN_N", "3072"))
_USE_CUDA = os.environ.get("SHAFT_DEVICE", "cuda") == "cuda" and torch.cuda.is_available()

_orig_matmul_forward = _ctmod.MatMul.forward


def _empty():
    gc.collect()
    if _USE_CUDA:
        torch.cuda.empty_cache()


def _chunked_matmul_forward(self, x):
    if hasattr(self, "weight"):
        a, b = x, self.weight
    else:
        if not (isinstance(x, (list, tuple)) and len(x) == 2):
            return _orig_matmul_forward(self, x)
        a, b = x[0], x[1]

    # Only chunk the heavy matmuls (large output dim); everything else runs stock.
    if _NCH <= 1 or b.dim() < 2 or b.shape[-1] < _MIN_N:
        return _orig_matmul_forward(self, x)

    n = b.shape[-1]
    cs = (n + _NCH - 1) // _NCH
    parts = []
    for s in range(0, n, cs):
        e = min(s + cs, n)
        parts.append(a.matmul(b[..., s:e]))
        _empty()
    out = crypten.cat(parts, dim=-1)
    del parts
    _empty()
    return out


_ctmod.MatMul.forward = _chunked_matmul_forward
print(f"matmul_chunk_gpu: MatMul.forward chunked "
      f"(chunks={_NCH}, min_out_dim={_MIN_N}, cuda={_USE_CUDA})",
      file=sys.stderr, flush=True)
