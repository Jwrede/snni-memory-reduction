"""Diagnostic only. Referrers of the ~143 MB plaintext float32 activation tensors alive during
encrypted inference (not owned by the CrypTen graph, probe_graph_owners_cpu.py). Walks referrers two
levels from a sample (gc.get_referrers is expensive). Never in a measured run.
"""

import gc
import os
import sys

import torch

from crypten.nn import module as cnnmod

_ORIG = cnnmod.Graph.forward
_DONE = False
_SHAPES = [(1, 128, 768), (1, 128, 3072), (1, 12, 128, 128)]


def _describe(o, depth=0):
    t = type(o)
    s = f"{t.__module__}.{t.__name__}"
    if isinstance(o, dict):
        keys = [k for k in list(o.keys())[:6] if isinstance(k, str)]
        s += f" dict(keys~{keys})"
    elif isinstance(o, (list, tuple)):
        s += f" len={len(o)}"
    elif hasattr(o, "__class__") and hasattr(o, "__dict__"):
        s += " obj"
    try:
        import types
        if isinstance(o, types.FrameType):
            s = f"FRAME {o.f_code.co_filename}:{o.f_lineno}:{o.f_code.co_name}"
    except Exception:
        pass
    return s


def forward(self, *args):
    global _DONE
    if not _DONE:
        _DONE = True
        out = open(os.path.join(os.environ.get("RESULTS_DIR", "/tmp"),
                                f"fp32refs_pid{os.getpid()}.txt"), "w", buffering=1)
        for shape in _SHAPES:
            # Never build a list of the matches: the first version collected them all, then every
            # referrer answer was its own collection list, so the probe measured itself.
            n = 0
            sample = []
            for obj in gc.get_objects():
                try:
                    if (isinstance(obj, torch.Tensor) and obj.dtype is torch.float32
                            and tuple(obj.shape) == shape):
                        n += 1
                        if len(sample) < 3:
                            sample.append(obj)
                except Exception:
                    continue
            print(f"FP32REFS|shape={shape}|count={n}", file=out, flush=True)
            for t in sample:
                shown = 0
                for r in gc.get_referrers(t):
                    if r is sample or r is t:
                        continue
                    print(f"    L1 {_describe(r)}", file=out, flush=True)
                    for r2 in gc.get_referrers(r):
                        if r2 is sample or r2 is r:
                            continue
                        print(f"        L2 {_describe(r2)}", file=out, flush=True)
                        shown += 1
                        if shown > 8:
                            break
                    if shown > 8:
                        break
                print("    ---", file=out, flush=True)
            del sample
        out.close()
    return _ORIG(self, *args)


print(f"LEVER|probe_fp32_refs_cpu|patched pid={os.getpid()}", file=sys.stderr, flush=True)
cnnmod.Graph.forward = forward
