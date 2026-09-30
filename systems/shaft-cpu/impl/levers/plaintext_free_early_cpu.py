"""Plaintext model released before the ONNX bytes are parsed.
Supersedes plaintext_free_cpu.py (both replace crypten.nn.from_pytorch); use one. After s3 the peak is
two humps within 0.7% (two of three replicates inside from_pytorch, one in the embedding). During
from_onnx: plaintext model (433250304 B), ONNX buffer, parsed protobuf and graph at once; release one
statement earlier removes 433250304 B. Free (bytes complete first; training flag captured first).
"""

import sys

import torch
import crypten.nn as cnn
from crypten.nn import onnx_converter


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
    crypten_model = onnx_converter.from_onnx(f)
    f.close()
    crypten_model.pytorch_model = None
    crypten_model.train(mode=training)
    print(f"LEVER|plaintext_free_early_cpu|released_plaintext_bytes={freed}",
          file=sys.stderr, flush=True)
    return crypten_model


onnx_converter.from_pytorch = from_pytorch
cnn.from_pytorch = from_pytorch

print("LEVER|plaintext_free_early_cpu|patched", file=sys.stderr, flush=True)
