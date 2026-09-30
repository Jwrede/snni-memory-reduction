"""Trace the model for ONNX export under `no_grad`, so the dummy forward keeps no autograd graph.

THE OBJECT: 143130624 B of plaintext float32 tensors with ACTIVATION shapes, alive from the model
conversion through the whole encrypted inference. 100 of (1, 128, 768), 48 of (1, 128, 3072), 24 of
(1, 12, 128, 128), 24 of (1, 128, 12, 64) -- one forward pass's worth of BERT-base intermediates.

HOW IT WAS IDENTIFIED, and the two wrong answers that were ruled out first:

  - NOT folded ONNX constants. `_export_pytorch_model` sets `do_constant_folding: False` and then
    `kwargs.update(default_kwargs)`, so the default WINS over anything a caller passes
    (`nn/onnx_converter.py:141,150`). Independently, the graph's `Constant` modules hold 3564 B in
    total (`probe_graph_owners_cpu.py`).
  - NOT reachable from Python. `gc.get_referrers` finds nothing referring to any of them
    (`probe_fp32_refs_cpu.py`), so no Python object owns them.

What is left is a C++ owner, and the shapes say which one: exactly one forward pass of intermediates,
which is what an autograd graph retains. `torch.onnx.export` traces by RUNNING the model, `model.eval()`
does not disable gradient recording, and every intermediate is then held by the `grad_fn` chain of the
traced output -- C++ nodes, invisible to `gc.get_referrers`, which is why the referrer probe came back
empty rather than inconclusive.

THE FIX IS A SOURCE CHANGE, not a bind-mounted trick: wrap the export in `torch.no_grad()`. With
gradient recording off, no graph is built and no intermediate is saved for backward.

WHY IT IS FREE. Not building an autograd graph is strictly less work. Nothing is recomputed, nothing is
reloaded.

WHY THE GATE CANNOT MOVE. The exported ONNX carries the same operators and the same initializers; the
trace is a shape-and-op recording, and gradient bookkeeping is not part of it. No generator is
consulted: the model is in eval mode, so its 38 Dropout modules are identities and draw nothing.
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
