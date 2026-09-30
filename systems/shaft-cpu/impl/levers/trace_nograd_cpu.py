"""ONNX-export trace under no_grad (the dummy forward keeps no autograd graph).
Object: 143130624 B of plaintext float32 activation-shaped tensors (one forward of BERT-base
intermediates), alive from conversion through inference. Excluded: folded constants
(_export_pytorch_model forces do_constant_folding False; Constant modules 3564 B) and Python owners
(gc.get_referrers empty, probe_fp32_refs_cpu.py). Owner: C++ grad_fn chain of the traced output
(model.eval() does not disable gradient recording). Fix: export inside torch.no_grad(). Free; gate
unchanged (same operators and initializers; eval-mode Dropout draws nothing).
"""

import sys

import torch

from crypten.nn import onnx_converter

_ORIG = onnx_converter._export_pytorch_model


def _export_pytorch_model(f, pytorch_model, dummy_input, **kwargs):
    # Both export calls in `_from_pytorch_to_bytes` go through this name, so patching it here covers
    # the discarded first export as well as the real second one.
    with torch.no_grad():
        return _ORIG(f, pytorch_model, dummy_input, **kwargs)


onnx_converter._export_pytorch_model = _export_pytorch_model

print("LEVER|trace_nograd_cpu|patched", file=sys.stderr, flush=True)
