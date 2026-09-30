#!/usr/bin/env python3
"""
SHAFT-CPU, per-layer variant: convert, encrypt, run and discard ONE sub-model at a time.

WHY THIS IS A SEPARATE SCRIPT AND NOT A LEVER. Every other reduction on this system is a monkeypatch
of CrypTen, loaded by name into the ONE measured script whose md5 is the identity of the measured
program. This one changes how the program DRIVES the model: instead of converting the whole network
once and running it, it converts the embedding, then each encoder layer, then the head, running and
freeing each before the next exists. Nothing in CrypTen can be patched to do that, because the
driving happens above CrypTen. `SHAFT_SCRIPT` selects it and `baseline_script_md5` in RUN_META
records that a different program ran, which is exactly what that field is for.

PROVENANCE. Ported from the retired campaign's `operator-bench/shaft/02_per_layer/
bert_instrumented.py`, whose sub-model wrappers and loop structure are reproduced here. That is the
step its 629 MiB endpoint (5.2x) rests on, and this campaign has never been able to admit it,
because that line had NO GATE AT ALL: its `ct.init()` carries no seeding call, CrypTen seeds from
`os.urandom(8)`, so its output differed run to run and no step was ever compared against a baseline.

WHAT IS DIFFERENT HERE, and all of it is the measurement rather than the lever:

  * the seeds are pinned exactly as in `bert_instrumented_cpu.py`, so an output comparison exists;
  * the decrypted logits are printed at full precision as `GATE|logits|...`;
  * the in-process VmRSS poller is removed, so one run produces one trace from one instrument;
  * the transcript counters are summed across the fourteen sub-graphs and printed as integers,
    because PROCEDURE's phase B rests on "the same bytes" and CrypTen's own two-decimal print has
    5 MB of resolution.

WHY IT NEEDS PHASE B, stated before the number arrives. Converting and encrypting layer i+1 while
layer i's result is already computed interleaves layer i+1's PRZS draws with layer i's Beaver draws.
The mask stream therefore differs from the baseline's, fixed-point truncation error is
share-dependent, and the exact gate cannot separate that from a broken protocol. This system has
measured how large that is: at twelve layers a changed mask stream moves the observable by 2.34 on a
value of 0.135, and at one layer the argmax itself flips for one seed in five
(`GATE-TOLERANCE.md`). So no tolerance can admit this step and the two phase B conditions have to
carry it instead:

  (a) THE TRANSCRIPT. The same bytes must cross the wire. That is what this script's summed
      counters are for, and it is the condition the retired campaign never checked.
  (b) THE PLAINTEXT EQUIVALENCE. The decomposition must compute the same function. Run outside the
      protocol this is exact: `impl/equiv_per_layer.py` compares whole-model against per-layer
      inference in plaintext.

THE ATTENTION MASK, and this is the one place where the port is not literal. The whole-model graph
takes `attention_mask` as an input and computes the extended mask inside. Split apart, that
computation has no home, so the extended mask is built directly as
`ct.cryptensor(torch.zeros(1, 1, 1, max_length))`. **That is equivalent for this workload and only
for this workload**: the measured configuration feeds `attention_mask = torch.ones(...)`, and an
all-ones mask produces an all-zeros extended mask (nothing is masked out). The retired campaign did
the same. If the workload ever gains padding, this script becomes wrong and silently so, which is
why it is written down here rather than left to be noticed.

Marker format on stderr, unchanged:
    MEM|<party>|<layer>|<phase>|<op>|<rss_kb>|<timestamp_ms>
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


# --------------- Sub-model wrappers for ONNX tracing ---------------
# Reproduced from the retired campaign's 02_per_layer. Each exists because `from_pytorch` traces a
# module with a fixed signature, and a bare `BertLayer` returns a tuple.

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
    """`from_pytorch` keeps a reference to the traced module on the graph it returns.

    Left in place it would defeat the whole point: the plaintext sub-model would stay alive for as
    long as its encrypted counterpart. The retired campaign deleted it too.
    """
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

    # DETERMINISTIC SEEDING, verbatim from `bert_instrumented_cpu.py`. It is what makes this
    # variant comparable at all, and its absence is the single defect that makes the retired
    # campaign's version of this step unverifiable after the fact.
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

    # ---- Take the sub-models apart and drop the container ----
    # The list entries are replaced with placeholders so `del model` frees everything that is not
    # one of the pieces below, rather than leaving the whole graph alive through one reference.
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

    # ---- Encrypt inputs ----
    # Same workload as the baseline: the same vocabulary bound, the same length, the same
    # all-ones attention mask (here already in its extended, all-zeros form; see the header).
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

    # THE TRANSCRIPT, summed over the fourteen sub-graphs and printed as integers. The baseline
    # reads the same attribute from its single graph, so the two numbers are the same quantity
    # measured over the same protocol; what phase B asks is whether they are EQUAL.
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
