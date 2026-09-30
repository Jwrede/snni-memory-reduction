#!/usr/bin/env python3
"""Plaintext equivalence test for the per-layer decomposition (PROCEDURE.md phase B, condition (b)).
bert_perlayer_cpu.py: chain of 14 traced sub-models. Checks: chain vs whole model (same weights and
input; expected zero difference); extended mask built as zeros vs get_extended_attention_mask
(all-ones workload; false with padding). Arithmetic only.
Run inside the measured image: python3 equiv_per_layer.py (exits non-zero on any mismatch).
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

    # Index range from the model, not the script: the measured scripts draw randint(0, 30522,...)
    # (bert-base-UNCASED); this model is CASED with 28,996 entries. Harmless in the protocol (one-hot
    # matmul, index reduced % voc_size), but torch.embedding raises in plaintext, which is how it was
    # noticed. Valid indices used here since this tests whole-vs-decomposition, not the index range.
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
        # NOT silently tolerated: same modules, same order, so a difference means the wiring differs.
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
