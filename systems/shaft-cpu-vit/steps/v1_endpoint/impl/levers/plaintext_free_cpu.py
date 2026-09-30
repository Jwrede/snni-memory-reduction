"""s1_plaintext_free: stop holding the plaintext model after it has been encrypted.

CHOSEN FROM THE MEASURED DECOMPOSITION OF s0_default's PEAK (3594240 kB), not inherited.

WHAT THE MEASUREMENT SAYS. At the peak the process holds 866500608 B of plaintext float32 model
WEIGHTS, which is exactly `2 x 433250304`: two complete copies of BERT-base in the clear, sitting
beside the int64 secret shares that the computation actually uses. It is the largest object at that
peak (the next is the int64 word-embedding share group, 712605696 B). Evidence:
`steps/s0_default/evidence/README.md`, with the census time series showing the float32 (768, 768)
group stepping from 49 tensors to 98 at the moment `from_pytorch` runs, and staying at 98 through
the peak.

WHERE THE TWO COPIES COME FROM, from the source rather than from the sizes:

  1. `crypten/nn/onnx_converter.py:64`
         crypten_model.pytorch_model = copy.deepcopy(pytorch_model)
     A full deep copy of the plaintext model, attached to the returned CrypTen model. Its only
     consumer in the whole library is `Module.to_pytorch()` (`nn/module.py:924-933`), the
     export-the-model-back path, which this workload never calls.

  2. the caller's own reference. The measured script hands its `model` to `from_pytorch` and never
     reads it again: the CrypTen graph is built from the ONNX bytes and carries its own weights,
     and after `encrypt()` every value the protocol touches is an int64 share.

WHY THE FIX IS FREE. Both are retentions past last use, which is the first of PROCEDURE.md's three
conditions, and they are also duplicates of each other, which is the second. Releasing memory
performs no computation: nothing is recomputed, nothing is reloaded, no communication round is
added. The ONNX bytes are produced and `from_onnx` has already built the graph from them before
anything here is released, so the encrypted weights are complete and untouched.

WHY THE GATE CANNOT MOVE. The protocol reads only the int64 shares. The plaintext parameters are
not inputs to any operation after conversion, and no random number is drawn differently: this
module consumes no randomness and changes no loop bounds or thread partitioning. If the gate DOES
move, the claim above is wrong and the step is void -- which is the point of gating it.

NOT AN OFF-PATH TRADE: no accuracy, no security property and no portability is given up. The model
is a fixed public checkpoint here; what is released is a redundant copy of data the encrypted graph
already holds.
"""

import sys

import torch
import crypten.nn as cnn
from crypten.nn import onnx_converter

_ORIG_FROM_PYTORCH = onnx_converter.from_pytorch


def _release_parameters(module):
    """Drop the storage behind every parameter and buffer of a plaintext torch module.

    `p.data = torch.empty(0)` rather than `del`: the caller still holds a reference to the module
    object, so the module has to survive while its tensors do not. Anything that then tried to READ
    a weight would fail loudly on a zero-element tensor, which is the behaviour we want -- a silent
    zero would be a wrong answer, and this campaign has already published one step whose mechanism
    was a bug while its number looked right.
    """
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
    # Identical to crypten.nn.from_pytorch except for the two retentions. Reimplemented rather than
    # called-and-then-cleaned, because calling the original would MAKE the 433 MB deep copy first
    # and only then free it, which is a different program from one that never makes it.
    f = onnx_converter._from_pytorch_to_bytes(pytorch_model, dummy_input, **kwargs)
    crypten_model = onnx_converter.from_onnx(f)
    f.close()

    # (1) no deepcopy. The attribute stays present and empty so that to_pytorch() fails with its
    # own message rather than an AttributeError, if anything ever calls it.
    crypten_model.pytorch_model = None

    # train/eval must still be copied: it decides whether Dropout modules are active, which WOULD
    # change what is computed.
    crypten_model.train(mode=pytorch_model.training)

    # (2) release the caller's plaintext weights. Done after from_onnx has built the graph, so the
    # encrypted weights are already materialised from the ONNX initializers.
    freed = _release_parameters(pytorch_model)
    print(f"LEVER|plaintext_free_cpu|released_plaintext_bytes={freed}",
          file=sys.stderr, flush=True)
    return crypten_model


# Patch BOTH names. `crypten/nn/__init__.py:74` re-exports from_pytorch by value, so replacing it
# only in onnx_converter would leave the caller's `ct.nn.from_pytorch` bound to the original and
# the run would measure the baseline under this lever's name -- the exact failure the announcement
# check below exists to catch.
onnx_converter.from_pytorch = from_pytorch
cnn.from_pytorch = from_pytorch

# The announcement the harness greps for. It is printed here, at patch time, and only here: a
# marker that also printed in a no-op case would read as confirmation while proving nothing.
print("LEVER|plaintext_free_cpu|patched", file=sys.stderr, flush=True)
