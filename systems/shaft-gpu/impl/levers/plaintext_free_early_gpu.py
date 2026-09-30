"""s3_plaintext_free_early: release the plaintext model after _from_pytorch_to_bytes produces the
ONNX buffer and before from_onnx parses it. Supersedes plaintext_free_gpu (released after the parse,
past this system's peak, 7.4 s before the model reaches CUDA); both replace crypten.nn.from_pytorch,
use one. Free (dead weights, buffer complete before release); gate must stay byte-identical."""
import sys

import torch
import crypten.nn as cnn
from crypten.nn import onnx_converter

_ORIG_FROM_PYTORCH = onnx_converter.from_pytorch


def _release_parameters(module):
    """Drop the storage behind every parameter and buffer of a plaintext torch module.
    `p.data = torch.empty(0)` not `del`: the module must survive while its tensors do not, and a
    later read then fails loudly on a zero-element tensor rather than returning a silent zero."""
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
    # Reimplemented, not called-then-cleaned: calling the original would make the 433 MB deep copy
    # first and only then free it.
    f = onnx_converter._from_pytorch_to_bytes(pytorch_model, dummy_input, **kwargs)

    # The one change against plaintext_free_gpu: release here, not after the parse. `training` is
    # read first because `crypten_model.train(mode=...)` below still needs it.
    _training = pytorch_model.training
    freed = _release_parameters(pytorch_model)

    crypten_model = onnx_converter.from_onnx(f)
    f.close()

    # No deepcopy; the attribute stays present and empty so to_pytorch() fails with its own message.
    crypten_model.pytorch_model = None

    # train/eval must still be copied: it decides whether Dropout is active, which would change the result.
    crypten_model.train(mode=_training)

    print(f"LEVER|plaintext_free_early_gpu|released_plaintext_bytes={freed}",
          file=sys.stderr, flush=True)
    return crypten_model


# Patch BOTH names: crypten/nn/__init__.py re-exports from_pytorch by value, so patching only
# onnx_converter would leave the caller's ct.nn.from_pytorch on the original.
onnx_converter.from_pytorch = from_pytorch
cnn.from_pytorch = from_pytorch

print("plaintext_free_early_gpu: patched", file=sys.stderr, flush=True)
