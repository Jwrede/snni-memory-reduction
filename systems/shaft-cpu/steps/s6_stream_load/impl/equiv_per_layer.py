#!/usr/bin/env python3
"""Plaintext equivalence test, per-layer decomposition. PROCEDURE.md phase B, condition (b).

bert_perlayer_cpu.py replaces

    logits = model(input_ids, attention_mask, token_type_ids)

with fourteen independently traced sub-models

    h = EmbeddingWrapper(embeddings)(input_ids, token_type_ids)
    for i in range(12): h = EncoderLayerWrapper(layer_i)(h, extended_mask)
    logits = ClassifierHeadWrapper(pooler, classifier)(h)

Interleaved PRZS/Beaver draws change the protocol output, so the exact gate cannot certify the step;
the transcript covers protocol work, this test covers the function on plaintext.

Tests:
  1. Chain: whole model vs decomposition, same weights and input; expected difference zero (magnitude
     reported otherwise; detects residual-stream miswiring).
  2. Mask: the script builds the extended mask as torch.zeros(1, 1, 1, seq) for an all-ones
     attention_mask; checked against get_extended_attention_mask. Invalid with padding.
Not shown: equivalence of the secret-shared computation (follows from linearity; transcript checks it).

Run inside the measured image: python3 equiv_per_layer.py   (non-zero exit on mismatch)
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

    # THE INDEX RANGE IS THE MODEL'S, NOT THE SCRIPT'S, and the difference is worth recording.
    # Both measured scripts draw `torch.randint(0, 30522, ...)`, which is bert-base-UNCASED's
    # vocabulary; this model is bert-base-CASED with 28,996 entries. In the protocol that is
    # harmless because the private embedding is a one-hot matmul and the index is reduced
    # `% voc_size`, so an out-of-range draw silently wraps. In plaintext `torch.embedding` raises
    # instead, which is how this was noticed. The equivalence being tested here is between the
    # whole model and its decomposition, not about the workload's index range, so valid indices
    # are used and the discrepancy is written down rather than papered over.
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, seq))
    attention_mask = torch.ones(1, seq, dtype=torch.long)
    token_type_ids = torch.zeros(1, seq, dtype=torch.long)

    failures = []

    # ---- (2) the mask assumption, checked against the model's own method ----
    ext = model.bert.get_extended_attention_mask(attention_mask, (1, seq))
    assumed = torch.zeros(1, 1, 1, seq, dtype=ext.dtype)
    if ext.shape != assumed.shape:
        failures.append(f"extended mask shape {tuple(ext.shape)} != assumed {tuple(assumed.shape)}")
    elif not torch.equal(ext, assumed):
        failures.append(f"extended mask is not all zeros for an all-ones attention mask; "
                        f"max |value| = {ext.abs().max().item()}")
    else:
        print(f"  ok  extended mask for an all-ones attention_mask is all zeros, shape "
              f"{tuple(ext.shape)}")

    # ---- (1) whole model against the decomposition ----
    with torch.no_grad():
        whole = model(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids).logits

        embed = EmbeddingWrapper(model.bert.embeddings).eval()
        h = embed(input_ids, token_type_ids)
        for i in range(n_layers):
            h = EncoderLayerWrapper(model.bert.encoder.layer[i]).eval()(h, assumed)
        head = ClassifierHeadWrapper(model.bert.pooler, model.classifier).eval()
        chained = head(h)

    if whole.shape != chained.shape:
        failures.append(f"shape {tuple(chained.shape)} != {tuple(whole.shape)}")
    elif torch.equal(whole, chained):
        print(f"  ok  {n_layers} layers: decomposition is bit-identical to the whole model "
              f"{[f'{v:.10e}' for v in whole.flatten().tolist()]}")
    else:
        d = (whole - chained).abs().max().item()
        rel = d / max(whole.abs().max().item(), 1e-30)
        # NOT silently tolerated. Both paths run the same modules in the same order, so a
        # difference means the wiring differs, not that floating point is imprecise.
        failures.append(f"decomposition differs from the whole model: max abs {d:.6e}, "
                        f"max rel {rel:.6e}; whole={whole.flatten().tolist()} "
                        f"chained={chained.flatten().tolist()}")

    if failures:
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        sys.exit(f"{len(failures)} case(s) failed: the decomposition does not compute the same "
                 f"function as the whole model")
    print("PASS: the per-layer decomposition computes the same function, and the mask assumption "
          "holds for the measured workload")


if __name__ == "__main__":
    main()
