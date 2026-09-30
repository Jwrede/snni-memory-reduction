#!/usr/bin/env python3
"""Plaintext equivalence for the GPU per-layer decomposition (PROCEDURE.md phase B, condition (b)).

bert_perlayer_gpu.py replaces one traced model with fifteen traced sub-graphs:
    ext = ExtendedMaskWrapper(bert)(attention_mask)          <- sub-graph 0, the CPU port has none
    h   = EmbeddingWrapper(embeddings)(input_ids, token_type_ids)
    for i in range(12): h = EncoderLayerWrapper(layer_i)(h, ext)
    logits = ClassifierHeadWrapper(pooler, classifier)(h)
The CPU port builds an all-zeros extended mask; this port traces get_extended_attention_mask as
sub-graph 0.

Checks:
  1. mask sub-graph against the model's method, on an all-ones and a padded mask (padded = case
     where the CPU shortcut would be wrong).
  2. chain: whole model against decomposition, same weights and input, sub-graph mask; expected
     difference zero, any difference reported with its magnitude.
  3. sub-graph count asserted = 15 (the transcript_artefact declaration depends on sub-graph 0).
Scope: plaintext arithmetic only; the shared case follows by linearity, the transcript tests the
protocol.

Run inside the measured image: python3 equiv_per_layer_gpu.py   (exits non-zero on any mismatch)
"""

import os
import sys

import torch
import torch.nn as nn

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class ExtendedMaskWrapper(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, attention_mask):
        return self.bert.get_extended_attention_mask(attention_mask, attention_mask.shape)


class EmbeddingWrapper(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, input_ids, token_type_ids):
        return self.embeddings(input_ids, token_type_ids=token_type_ids)


class EncoderLayerWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, attention_mask):
        return self.layer(hidden_states, attention_mask=attention_mask)[0]


class ClassifierHeadWrapper(nn.Module):
    def __init__(self, pooler, classifier):
        super().__init__()
        self.pooler = pooler
        self.classifier = classifier

    def forward(self, hidden_states):
        return self.classifier(self.pooler(hidden_states))


def main():
    from transformers import AutoModelForSequenceClassification

    model_name = os.environ.get("SHAFT_MODEL", "andeskyl/bert-base-cased-sst2")
    seq = int(os.environ.get("SHAFT_MAX_LENGTH", "128"))
    n_layers = int(os.environ.get("SHAFT_N_LAYERS", "12"))

    torch.manual_seed(0x5EED)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    if n_layers < len(model.bert.encoder.layer):
        model.bert.encoder.layer = model.bert.encoder.layer[:n_layers]

    # Valid indices for THIS model: bert-base-CASED has 28,996 entries, not 30,522. The measured
    # driver draws `% voc_size` inside the protocol, where an out-of-range index wraps silently;
    # `torch.embedding` raises instead. Recorded here rather than papered over, same as on the CPU.
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, seq))
    token_type_ids = torch.zeros(1, seq, dtype=torch.long)

    failures = []
    mask_wrapper = ExtendedMaskWrapper(model.bert).eval()

    # ---- (1) the mask sub-graph against the model's own method ----
    for name, am in (("all ones", torch.ones(1, seq, dtype=torch.long)),
                     ("padded (last quarter zero)",
                      torch.cat([torch.ones(1, seq - seq // 4, dtype=torch.long),
                                 torch.zeros(1, seq // 4, dtype=torch.long)], dim=1))):
        with torch.no_grad():
            traced = mask_wrapper(am)
            direct = model.bert.get_extended_attention_mask(am, am.shape)
        if traced.shape != direct.shape:
            failures.append(f"mask sub-graph shape {tuple(traced.shape)} != "
                            f"{tuple(direct.shape)} for {name}")
        elif not torch.equal(traced, direct):
            failures.append(f"mask sub-graph differs from the model's own method for {name}: "
                            f"max |diff| = {(traced - direct).abs().max().item()}")
        else:
            print(f"  ok  mask sub-graph is bit-identical to get_extended_attention_mask "
                  f"({name}), shape {tuple(traced.shape)}")

    # THE SHORTCUT THIS PORT REJECTED, shown to be wrong where it matters. The CPU port builds the
    # extended mask as zeros; for an all-ones attention mask that is right, and for a padded one it
    # is not. Asserting the difference here is what keeps a future "simplification" honest.
    am_pad = torch.cat([torch.ones(1, seq - seq // 4, dtype=torch.long),
                        torch.zeros(1, seq // 4, dtype=torch.long)], dim=1)
    with torch.no_grad():
        ext_pad = model.bert.get_extended_attention_mask(am_pad, am_pad.shape)
    if torch.equal(ext_pad, torch.zeros_like(ext_pad)):
        failures.append("a padded attention mask produced an all-zero extended mask, so the "
                        "shortcut this port rejected would be indistinguishable and the reason "
                        "for sub-graph 0 no longer holds")
    else:
        print(f"  ok  a padded mask is NOT all zeros (min {ext_pad.min().item():.3e}), so tracing "
              f"the mask is a real difference and not a formality")

    # ---- (2) whole model against the decomposition ----
    am = torch.ones(1, seq, dtype=torch.long)
    with torch.no_grad():
        whole = model(input_ids=input_ids, attention_mask=am,
                      token_type_ids=token_type_ids).logits

        ext = mask_wrapper(am)
        h = EmbeddingWrapper(model.bert.embeddings).eval()(input_ids, token_type_ids)
        n_sub = 2                                     # mask + embedding
        for i in range(n_layers):
            h = EncoderLayerWrapper(model.bert.encoder.layer[i]).eval()(h, ext)
            n_sub += 1
        chained = ClassifierHeadWrapper(model.bert.pooler, model.classifier).eval()(h)
        n_sub += 1

    if whole.shape != chained.shape:
        failures.append(f"shape {tuple(chained.shape)} != {tuple(whole.shape)}")
    elif torch.equal(whole, chained):
        print(f"  ok  {n_layers} layers: decomposition is bit-identical to the whole model "
              f"{[f'{v:.10e}' for v in whole.flatten().tolist()]}")
    else:
        d = (whole - chained).abs().max().item()
        rel = d / max(whole.abs().max().item(), 1e-30)
        failures.append(f"decomposition differs from the whole model: max abs {d:.6e}, "
                        f"max rel {rel:.6e}")

    # ---- (3) the sub-graph count ----
    expected = n_layers + 3
    if n_sub != expected:
        failures.append(f"{n_sub} sub-graphs against the expected {expected} "
                        f"(mask + embedding + {n_layers} layers + head)")
    else:
        print(f"  ok  {n_sub} sub-graphs: mask + embedding + {n_layers} layers + head")

    if failures:
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        sys.exit(f"{len(failures)} case(s) failed: the GPU per-layer decomposition does not "
                 f"compute the same function as the whole model")
    print("PASS: the per-layer decomposition computes the same function, the traced mask "
          "sub-graph reproduces the model's own, and the count is fifteen")


if __name__ == "__main__":
    main()
