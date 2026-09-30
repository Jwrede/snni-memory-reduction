#!/usr/bin/env python3
"""Plaintext equivalence test for the ViT per-layer decomposition (PROCEDURE.md phase B, (b)).
Separate from equiv_per_layer.py (evidence is model-specific; GEOMETRY-TRANSFER.md).
vit_perlayer_stream_cpu.py: chain of 14 traced sub-models. Checks: chain vs whole model (expected
zero difference); head order classifier(layernorm(seq)[:,0,:]) vs the model's forward (ViT has no
attention mask). Arithmetic only.
Run inside the measured image: python3 equiv_per_layer_vit.py (exits non-zero on any mismatch).
"""

import os
import sys

import torch
import torch.nn as nn

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class EmbeddingWrapper(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, pixel_values):
        return self.embeddings(pixel_values)


class EncoderLayerWrapper(nn.Module):
    """No mask argument: `ViTLayer.forward(hidden_states, head_mask=None, ...)`."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states):
        return self.layer(hidden_states)[0]


class ClassifierHeadWrapper(nn.Module):
    """Final LayerNorm, THEN the CLS slice, then the classifier. The order is the point."""

    def __init__(self, layernorm, classifier):
        super().__init__()
        self.layernorm = layernorm
        self.classifier = classifier

    def forward(self, hidden_states):
        return self.classifier(self.layernorm(hidden_states)[:, 0, :])


def main():
    from transformers import AutoModelForImageClassification

    model_name = os.environ.get("SHAFT_MODEL", "google/vit-base-patch16-224")
    n_layers = int(os.environ.get("SHAFT_N_LAYERS", "12"))

    torch.manual_seed(0x5EED)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    if n_layers < len(model.vit.encoder.layer):
        model.vit.encoder.layer = model.vit.encoder.layer[:n_layers]

    # Input shape from the model's config, not the script: ViTEmbeddings asserts on the image size
    # rather than wrapping, so a different patch grid is measured correctly instead of failing here.
    cfg = model.config
    px = torch.randn(1, cfg.num_channels, cfg.image_size, cfg.image_size)

    failures = []

    # ---- (2) the head, checked against the model's own forward ----
    # Probe takes the ENCODER's output, not ViTModel's: ViTModel.forward already applies self.layernorm
    # before returning, so model.vit(px)[0] would normalise twice (a 4.09e-01 false mismatch; the
    # first version had exactly this wiring bug).
    with torch.no_grad():
        enc = model.vit.encoder(model.vit.embeddings(px))[0]
        after_ln_then_cls = model.classifier(model.vit.layernorm(enc)[:, 0, :])
        after_cls_then_ln = model.classifier(model.vit.layernorm(enc[:, 0, :]))
        whole_probe = model(pixel_values=px).logits

    if torch.equal(whole_probe, after_ln_then_cls):
        print("  ok  head order is layernorm -> CLS slice -> classifier, as the driver assumes")
    elif torch.equal(whole_probe, after_cls_then_ln):
        failures.append("the model applies the layernorm AFTER the CLS slice; the driver's "
                        "ClassifierHeadWrapper has the two the wrong way round")
    else:
        d = (whole_probe - after_ln_then_cls).abs().max().item()
        failures.append(f"neither head order reproduces the model's logits; layernorm-first is off "
                        f"by max abs {d:.6e}, so the head is not what either reading assumed")

    # ---- (1) whole model against the decomposition ----
    with torch.no_grad():
        whole = model(pixel_values=px).logits

        h = EmbeddingWrapper(model.vit.embeddings).eval()(px)
        for i in range(n_layers):
            h = EncoderLayerWrapper(model.vit.encoder.layer[i]).eval()(h)
        chained = ClassifierHeadWrapper(model.vit.layernorm, model.classifier).eval()(h)

    if whole.shape != chained.shape:
        failures.append(f"shape {tuple(chained.shape)} != {tuple(whole.shape)}")
    elif torch.equal(whole, chained):
        print(f"  ok  {n_layers} layers, seq {h.shape[1]}: decomposition is bit-identical to the "
              f"whole model (max |logit| {whole.abs().max().item():.6e})")
    else:
        d = (whole - chained).abs().max().item()
        rel = d / max(whole.abs().max().item(), 1e-30)
        # NOT silently tolerated: same modules, same order, so a difference means the wiring differs.
        failures.append(f"decomposition differs from the whole model: max abs {d:.6e}, "
                        f"max rel {rel:.6e}")

    # Token count reported not assumed: it is the geometry probes' number, and a silent mismatch
    # would make two studies incomparable.
    print(f"  info  patch grid gives {h.shape[1]} tokens "
          f"({(cfg.image_size // cfg.patch_size) ** 2} patches + 1 CLS)")

    if failures:
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        sys.exit(f"{len(failures)} case(s) failed: the ViT decomposition does not compute the same "
                 f"function as the whole model")
    print("PASS: the ViT per-layer decomposition computes the same function, and the head order "
          "is the model's own")


if __name__ == "__main__":
    main()
