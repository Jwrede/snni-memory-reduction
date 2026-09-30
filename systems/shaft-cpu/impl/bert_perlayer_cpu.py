#!/usr/bin/env python3
"""SHAFT-CPU per-layer driver: convert, encrypt, run and discard one sub-model at a time.
Separate script (SHAFT_SCRIPT; baseline_script_md5 records it). Port of the retired 02_per_layer
(629 MiB, 5.2x, no gate). Measurement changes: seeds pinned, GATE|logits| at full precision, poller
removed, transcript summed over 14 sub-graphs as integers. Phase B (transcript + equiv_per_layer.py;
GATE-TOLERANCE.md): interleaved per-layer PRZS/Beaver draws change the mask stream. Extended attention
mask built as zeros (valid only for the all-ones mask; wrong with padding).
Stderr marker: MEM|<party>|<layer>|<phase>|<op>|<rss_kb>|<timestamp_ms>.
"""

import gc
import importlib
import os
import sys
import time

import torch
import torch.nn as nn

import crypten
import crypten as ct

for _lv in filter(None, (m.strip() for m in os.environ.get("SHAFT_LEVERS", "").split(","))):
    importlib.import_module(_lv)


def read_vmrss_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        return -1
    return -1


def mem(phase, op, layer_idx=-1):
    rss = read_vmrss_kb()
    ts = int(time.time() * 1000)
    rank = int(os.environ.get("RANK", -1))
    print(f"MEM|{rank}|{layer_idx}|{phase}|{op}|{rss}|{ts}", file=sys.stderr, flush=True)


# Sub-model wrappers for ONNX tracing (from 02_per_layer): from_pytorch traces a fixed signature,
# and a bare BertLayer returns a tuple.

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
        outputs = self.layer(hidden_states, attention_mask=attention_mask)
        return outputs[0]


class ClassifierHeadWrapper(nn.Module):
    def __init__(self, pooler, classifier):
        super().__init__()
        self.pooler = pooler
        self.classifier = classifier

    def forward(self, hidden_states):
        pooled = self.pooler(hidden_states)
        return self.classifier(pooled)


def _drop_pytorch_model(g):
    """Drop the traced module `from_pytorch` keeps on the returned graph; else the plaintext
    sub-model stays alive as long as its encrypted counterpart."""
    if hasattr(g, "pytorch_model"):
        del g.pytorch_model


def _comm_of(g):
    """(bytes, rounds) for one sub-graph, from the same attribute the baseline reads."""
    tb = getattr(g, "total_comm_bytes", None)
    tr = getattr(g, "total_comm_rounds", None)
    if tb is None:
        for m in getattr(g, "_modules", {}).values():
            if hasattr(m, "total_comm_bytes"):
                return m.total_comm_bytes, m.total_comm_rounds
        return 0, 0
    return tb, tr


def main():
    ct.init()
    device = "cpu"
    rank = ct.comm.get().get_rank()

    # Deterministic seeding, verbatim from bert_instrumented_cpu.py; makes this variant comparable.
    from crypten.config import cfg as _cfg
    _dbg = _cfg.debug.debug_mode
    _cfg.debug.debug_mode = True
    ct.manual_seed(0xDEADBEEF + rank, 0xC0FFEE + rank, 0x5EED)
    _cfg.debug.debug_mode = _dbg
    torch.manual_seed(0x5EED)
    print(f"SEEDED|party={rank}|next=0x{0xDEADBEEF + rank:x}|"
          f"local=0x{0xC0FFEE + rank:x}|global=0x5eed|torch=0x5eed",
          file=sys.stderr, flush=True)

    pid = os.getpid()
    results_dir = os.environ.get("RESULTS_DIR", "/results/shaft")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"party{rank}.pid"), "w") as f:
        f.write(str(pid))

    model_name = os.environ.get("SHAFT_MODEL", "andeskyl/bert-base-cased-sst2")
    max_length = int(os.environ.get("SHAFT_MAX_LENGTH", "128"))
    n_layers = int(os.environ.get("SHAFT_N_LAYERS", "12"))
    print(f"CONFIG|party={rank}|n_layers={n_layers}|seq_len={max_length}|model={model_name}",
          file=sys.stderr, flush=True)

    mem("main", "after_init")

    # ---- Load plaintext model ----
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        actual_layers = len(model.bert.encoder.layer)
        if n_layers < actual_layers:
            model.bert.encoder.layer = model.bert.encoder.layer[:n_layers]
            print(f"Truncated model to {n_layers} layers (was {actual_layers})",
                  file=sys.stderr, flush=True)

    mem("main", "after_model_load")

    # Take sub-models apart, replacing entries with placeholders so `del model` frees everything
    # that is not one of the pieces below.
    embeddings = model.bert.embeddings
    encoder_layers = list(model.bert.encoder.layer)
    pooler = model.bert.pooler
    classifier = model.classifier

    model.bert.embeddings = nn.Identity()
    model.bert.encoder.layer = nn.ModuleList()
    model.bert.pooler = nn.Identity()
    model.classifier = nn.Identity()
    del model
    gc.collect()
    mem("main", "after_extract_submodules")

    # ---- Encrypt inputs ----  (same workload as baseline; mask already in extended all-zeros form)
    input_ids = torch.randint(0, 28996, (1, max_length))
    token_type_ids = torch.zeros(1, max_length, dtype=torch.long)

    input_ids_enc = ct.cryptensor(input_ids).to(device)
    token_type_enc = ct.cryptensor(token_type_ids).to(device)
    ext_mask_enc = ct.cryptensor(torch.zeros(1, 1, 1, max_length)).to(device)

    del input_ids, token_type_ids
    gc.collect()
    mem("main", "after_input_encrypt")

    comm_bytes, comm_rounds = 0, 0
    t0 = time.time()

    # ==== Phase 1: embeddings ====
    embed_wrapper = EmbeddingWrapper(embeddings)
    embed_wrapper.eval()
    dummy_ids = torch.zeros(1, max_length, dtype=torch.long)
    dummy_tt = torch.zeros(1, max_length, dtype=torch.long)

    enc_embed = ct.nn.from_pytorch(embed_wrapper, (dummy_ids, dummy_tt))
    del embed_wrapper, embeddings, dummy_ids, dummy_tt
    _drop_pytorch_model(enc_embed)
    gc.collect()
    mem("embed", "after_convert")

    enc_embed.encrypt()
    gc.collect()
    mem("embed", "after_encrypt")

    with ct.no_grad():
        hidden = enc_embed(input_ids_enc, token_type_enc)

    b, r = _comm_of(enc_embed)
    comm_bytes += b or 0
    comm_rounds += r or 0
    del enc_embed, input_ids_enc, token_type_enc
    gc.collect()
    mem("embed", "after_run")

    # ==== Phase 2: encoder layers, one at a time ====
    dummy_hidden = torch.randn(1, max_length, 768)
    dummy_mask = torch.zeros(1, 1, 1, max_length)

    for i in range(n_layers):
        layer = encoder_layers[i]
        encoder_layers[i] = None          # the list must not keep it alive

        wrapper = EncoderLayerWrapper(layer)
        wrapper.eval()
        enc_layer = ct.nn.from_pytorch(wrapper, (dummy_hidden, dummy_mask))
        del wrapper, layer
        _drop_pytorch_model(enc_layer)
        gc.collect()

        enc_layer.encrypt()
        gc.collect()
        mem("layer", f"before_run_{i}", i)

        with ct.no_grad():
            hidden = enc_layer(hidden, ext_mask_enc)

        b, r = _comm_of(enc_layer)
        comm_bytes += b or 0
        comm_rounds += r or 0
        del enc_layer
        gc.collect()
        mem("layer", f"after_run_{i}", i)

    del ext_mask_enc, dummy_hidden, dummy_mask, encoder_layers
    gc.collect()

    # ==== Phase 3: pooler and classifier ====
    head_wrapper = ClassifierHeadWrapper(pooler, classifier)
    head_wrapper.eval()
    dummy_head = torch.randn(1, max_length, 768)

    enc_head = ct.nn.from_pytorch(head_wrapper, dummy_head)
    del head_wrapper, pooler, classifier, dummy_head
    _drop_pytorch_model(enc_head)
    gc.collect()

    enc_head.encrypt()
    gc.collect()

    with ct.no_grad():
        logits = enc_head(hidden)

    b, r = _comm_of(enc_head)
    comm_bytes += b or 0
    comm_rounds += r or 0
    del enc_head, hidden
    gc.collect()

    t1 = time.time()
    mem("main", "after_inference")

    # ---- Decrypt ----
    outputs = logits.get_plain_text()
    del logits
    gc.collect()
    mem("main", "after_decrypt")

    # Transcript summed over the 14 sub-graphs as integers; phase B asks whether it EQUALS the
    # baseline's single-graph count over the same protocol.
    print(f"TRANSCRIPT|party={rank}|bytes={comm_bytes}|rounds={comm_rounds}|subgraphs={n_layers + 2}",
          flush=True)

    if rank == 0:
        predictions = outputs.argmax(dim=-1)
        print(f"Prediction: {predictions.item()}", flush=True)
        gate = " ".join(f"{v:.10e}" for v in outputs.flatten().tolist())
        print(f"GATE|logits|{gate}", flush=True)
        print(f"Inference time: {t1 - t0:.2f}s", flush=True)

    mem("main", "after_finalize")


if __name__ == "__main__":
    from multiprocess_launcher import MultiProcessLauncher

    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()
