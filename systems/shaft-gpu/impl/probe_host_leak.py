#!/usr/bin/env python3
"""Diagnose the ~35 MB/layer host RSS accumulation in the per-layer loop (CPU-only). Loops
from_pytorch over BERT encoder layers logging VmRSS; PROBE_MODE selects the cleanup variant:
  baseline / nodeepcopy (skip the copy.deepcopy) / nodeepcopy_jit (+ clear onnx/jit tracer state)."""
import os
import sys
import gc
import ctypes

import torch
import crypten
import crypten as ct
import crypten.nn.onnx_converter as oc

libc = ctypes.CDLL("libc.so.6")
MODE = os.environ.get("PROBE_MODE", "baseline")


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def trim():
    gc.collect()
    libc.malloc_trim(0)


if MODE in ("nodeepcopy", "nodeepcopy_jit"):
    def _patched_from_pytorch(pytorch_model, dummy_input, **kwargs):
        f = oc._from_pytorch_to_bytes(pytorch_model, dummy_input, **kwargs)
        m = oc.from_onnx(f)
        f.close()
        m.train(mode=pytorch_model.training)
        return m  # NO crypten_model.pytorch_model deepcopy
    oc.from_pytorch = _patched_from_pytorch
    ct.nn.from_pytorch = _patched_from_pytorch
    crypten.nn.from_pytorch = _patched_from_pytorch


def clear_jit():
    try:
        torch._C._jit_clear_class_registry()
    except Exception:
        pass
    try:
        torch.jit._state._python_cu.drop_all_functions()
    except Exception:
        pass


class EncoderLayerWrapper(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, attention_mask):
        return self.layer(hidden_states, attention_mask=attention_mask)[0]


def main():
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("andeskyl/bert-base-cased-sst2")
    model.eval()
    layers = list(model.bert.encoder.layer)
    model.bert.encoder.layer = torch.nn.ModuleList()
    del model
    trim()

    dummy_hidden = torch.randn(1, 128, 768)
    dummy_mask = torch.zeros(1, 1, 1, 128)

    print(f"MODE={MODE} start_rss={rss_mb():.0f} MB", flush=True)
    for i in range(len(layers)):
        layer = layers[i]
        layers[i] = None
        wrapper = EncoderLayerWrapper(layer)
        wrapper.eval()
        enc = ct.nn.from_pytorch(wrapper, (dummy_hidden, dummy_mask))
        del wrapper, layer
        if hasattr(enc, "pytorch_model"):
            del enc.pytorch_model
        del enc
        gc.collect()
        libc.malloc_trim(0)
        if MODE == "nodeepcopy_jit":
            clear_jit()
        print(f"iter={i} rss={rss_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
