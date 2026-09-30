"""Diagnostic only. Owner of the ~143 MB float32 activation-shaped tensors alive after encryption:
tracing leftover (free fix) or folded constant of the all-ones mask (crypten.nn.Constant, not
encrypted; live input). Per module class: instance count and parameter/buffer bytes. Never in a
measured run.
"""

import os
import sys

import torch

from crypten.nn import module as cnnmod

_ORIG = cnnmod.Graph.forward
_DONE = False


def forward(self, *args):
    global _DONE
    if not _DONE:
        _DONE = True
        out = open(os.path.join(os.environ.get("RESULTS_DIR", "/tmp"),
                                f"graphowners_pid{os.getpid()}.txt"), "w", buffering=1)
        by_cls = {}
        for name, mod in self._modules.items():
            cls = type(mod).__name__
            n, nb, shapes = by_cls.get(cls, (0, 0, {}))
            n += 1
            for _, t in list(mod._parameters.items()) + list(mod._buffers.items()):
                if torch.is_tensor(t):
                    nb += t.numel() * t.element_size()
                    key = f"{tuple(t.shape)}/{t.dtype}"
                    shapes[key] = shapes.get(key, 0) + 1
                elif hasattr(t, "share") and torch.is_tensor(getattr(t, "share", None)):
                    s = t.share
                    nb += s.numel() * s.element_size()
                    key = f"{tuple(s.shape)}/{s.dtype}/SHARE"
                    shapes[key] = shapes.get(key, 0) + 1
            by_cls[cls] = (n, nb, shapes)
        print(f"GRAPHOWNERS|modules={len(self._modules)}", file=out, flush=True)
        for cls, (n, nb, shapes) in sorted(by_cls.items(), key=lambda kv: -kv[1][1]):
            print(f"{nb:>13d} B  n={n:<5d} {cls}", file=out, flush=True)
            for key, cnt in sorted(shapes.items(), key=lambda kv: -kv[1])[:8]:
                print(f"                    x{cnt:<5d} {key}", file=out, flush=True)
        out.close()
    return _ORIG(self, *args)


print(f"LEVER|probe_graph_owners_cpu|patched pid={os.getpid()}", file=sys.stderr, flush=True)
cnnmod.Graph.forward = forward
