"""Plaintext model released before the ONNX bytes are parsed, and its pages returned.
plaintext_free_early_cpu alone (s6_plaintext_free_early, 17.08.) freed 433255432 B; peak unchanged
(arena kept the pages; later hump grew; cf. param_stream_cpu: 866 MB for -0.74%, anon:[heap]
383.6 -> 555.3 MB). Difference: malloc_trim(0) after the release (global MALLOC_TRIM_THRESHOLD_=0
refused 17.08.: heap -1,044 MB, anon +788 MB). Supersedes plaintext_free_cpu and plaintext_free_early_cpu
(all replace from_pytorch). Free.
"""


import ctypes
import sys

import torch
import crypten.nn as cnn
from crypten.nn import onnx_converter


try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def _trim():
    """Hand the freed pages back to the kernel instead of leaving them in glibc's arena."""
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


def _release_parameters(module):
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
    training = pytorch_model.training
    f = onnx_converter._from_pytorch_to_bytes(pytorch_model, dummy_input, **kwargs)
    # The only difference from plaintext_free_cpu: this line moves above from_onnx.
    freed = _release_parameters(pytorch_model)
    _trim()
    crypten_model = onnx_converter.from_onnx(f)
    f.close()
    crypten_model.pytorch_model = None
    crypten_model.train(mode=training)
    print(f"LEVER|plaintext_free_early_trim_cpu|released_plaintext_bytes={freed}",
          file=sys.stderr, flush=True)
    return crypten_model


onnx_converter.from_pytorch = from_pytorch
cnn.from_pytorch = from_pytorch

print("LEVER|plaintext_free_early_trim_cpu|patched", file=sys.stderr, flush=True)
