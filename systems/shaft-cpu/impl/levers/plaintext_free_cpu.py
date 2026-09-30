"""s1_plaintext_free: plaintext model released after encryption.
Basis: s0_default peak 3594240 kB; 866500608 B float32 weights = 2 x 433250304 (onnx_converter.py:64
deepcopy, consumed only by to_pytorch, never called; and the caller's reference), largest object.
Free (past last use, duplicated; ONNX bytes and graph exist first). Gate unchanged (protocol reads int64
shares only).
"""

import sys

import torch
import crypten.nn as cnn
from crypten.nn import onnx_converter

_ORIG_FROM_PYTORCH = onnx_converter.from_pytorch


def _release_parameters(module):
    """Drop the storage behind every parameter and buffer of a plaintext torch module.
    `p.data = torch.empty(0)` rather than `del` (the caller still holds the module, which must survive
    while its tensors do not); a later read then fails loudly on a zero-element tensor rather than
    silently returning a wrong zero."""
    freed = 0
    for p in module.parameters(recurse=True):
        freed += p.numel() * p.element_size()
        p.data = torch.empty(0, dtype=p.dtype)
    for b in module.buffers(recurse=True):
        if torch.is_tensor(b):
            freed += b.numel() * b.element_size()
            b.data = torch.empty(0, dtype=b.dtype)
    return freed


def from_pytorch(pytorch_model, dummy_input, **kwargs):
    # Reimplemented (not called-and-cleaned) because calling the original would MAKE the 433 MB deep
    # copy before freeing it, a different program from one that never makes it.
    f = onnx_converter._from_pytorch_to_bytes(pytorch_model, dummy_input, **kwargs)
    crypten_model = onnx_converter.from_onnx(f)
    f.close()

    # (1) no deepcopy; attribute stays present and empty so to_pytorch() fails with its own message.
    crypten_model.pytorch_model = None

    # train/eval must still be copied: it decides whether Dropout modules are active.
    crypten_model.train(mode=pytorch_model.training)

    # (2) release the caller's plaintext weights, after from_onnx has built the graph.
    freed = _release_parameters(pytorch_model)
    print(f"LEVER|plaintext_free_cpu|released_plaintext_bytes={freed}",
          file=sys.stderr, flush=True)
    return crypten_model


# Patch BOTH names: crypten/nn/__init__.py:74 re-exports from_pytorch by value, so patching only
# onnx_converter would leave ct.nn.from_pytorch bound to the original.
onnx_converter.from_pytorch = from_pytorch
cnn.from_pytorch = from_pytorch

# The announcement the harness greps for, printed at patch time and only here.
print("LEVER|plaintext_free_cpu|patched", file=sys.stderr, flush=True)
