#!/usr/bin/env python3
"""Plaintext equivalence test for the ViT per-layer decomposition (PROCEDURE.md phase B, (b)).

Compared:
    logits = model(pixel_values)
against the chain of fourteen traced sub-models
    h = EmbeddingWrapper(vit.embeddings)(pixel_values)
    for i in range(12): h = EncoderLayerWrapper(vit.encoder.layer[i])(h)
    logits = ClassifierHeadWrapper(vit.layernorm, classifier)(h)

Checks:
  1. chain: whole model against decomposition, same weights and input; expected difference zero,
     any difference reported with its magnitude.
  2. head: ViTForImageClassification.forward = classifier(layernorm(sequence_output)[:, 0, :]);
     the driver's order (LayerNorm before the CLS slice) is checked against the model's forward.
     ViT has no attention mask, so BERT's mask check has no counterpart.
Scope: plaintext arithmetic only. The shared computation follows by linearity of the sharing;
the transcript measurement tests the protocol.
Reason: the per-layer step changes the protocol output (PRZS and Beaver draws interleave
differently), so the exact gate cannot certify it.

Run inside the measured image: python3 equiv_per_layer_vit.py   (exits non-zero on any mismatch)
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

    # THE INPUT SHAPE IS THE MODEL'S, NOT THE SCRIPT'S. BERT's test had to record that the measured
    # driver draws indices from the wrong vocabulary; the analogous trap here is the image size,
    # because `ViTEmbeddings` asserts on it rather than wrapping. Taking it from the config means a
    # model with a different patch grid is measured correctly instead of failing here.
    cfg = model.config
    px = torch.randn(1, cfg.num_channels, cfg.image_size, cfg.image_size)

    failures = []

    # ---- (2) the head, checked against the model's own forward ----
    # `sequence_output[:, 0, :]` AFTER the layernorm. Both orders type-check and both produce
    # plausible logits, so this is checked rather than read.
    # THE PROBE TAKES THE ENCODER'S OUTPUT, NOT `ViTModel`'s, AND THE FIRST VERSION GOT THAT WRONG.
    # `ViTModel.forward` already applies `self.layernorm` before returning, so building the probe
    # from `model.vit(px)[0]` normalises twice and reports a 4.09e-01 mismatch against a driver that
    # is in fact bit-exact. The check written to catch a wiring error had the wiring error.
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
        # NOT silently tolerated. Both paths run the same modules in the same order, so a
        # difference means the wiring differs, not that floating point is imprecise.
        failures.append(f"decomposition differs from the whole model: max abs {d:.6e}, "
                        f"max rel {rel:.6e}")

    # The token count is reported rather than assumed, because it is the number the geometry probes
    # on the other systems are set to and a silent mismatch would make two studies incomparable.
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
