#!/usr/bin/env python3
"""SHAFT-GPU per-layer variant: convert, encrypt, move to the device, run and discard ONE sub-model
at a time. Attacks the host pool (untouched by the four VRAM levers), whose peak is the whole-model
ONNX conversion. Ported from SHAFT-CPU's s6_per_layer; needs phase B (transcript + equivalence)."""

import gc
import os
import sys
import time

import torch
import torch.nn as nn

import crypten  # noqa: F401  (imported before the levers, exactly as the baseline does)
import graphop_markers_gpu  # noqa: F401  measurement-neutral per-node markers
import embed_chunk_gpu  # noqa: F401  chunks the word-embedding matmul (no-op unless CHUNKS>1)

# Env-gated lever imports, verbatim from the baseline: each lever patches CrypTen at import time and
# announces on stderr, imported only when its own env var asks. The declaration lists only the
# SHAFT_LEVERS modules, so dropping this block silently omits env-gated levers (cost a run, 18.08.).
if os.environ.get("SHAFT_LIMB_LOOP", "0") == "1":
    import matmul_limb_loop_gpu  # noqa: F401
if int(os.environ.get("SHAFT_MATMUL_CHUNKS", "1")) > 1:
    import matmul_chunk_gpu  # noqa: F401
if os.environ.get("SHAFT_PLAINTEXT_FREE", "0") == "1":
    import plaintext_free_gpu  # noqa: F401

# GENERIC LEVER SLOT, the baseline's too: `SHAFT_LEVERS=mod_a,mod_b` imports those and nothing else
# does. A lever that never loaded is a successful run with a zero delta and no error.
for _lever in os.environ.get("SHAFT_LEVERS", "").split(","):
    _lever = _lever.strip()
    if _lever:
        __import__(_lever)

import crypten as ct  # noqa: E402

DEVICE = os.environ.get("SHAFT_DEVICE", "cuda")


def read_rss_hwm_kb():
    """Current VmRSS and the kernel's exact VmHWM high-water mark, from one read.
    VmHWM is monotone and exact, so it brackets this system's transient host peak (which the poller reads low)."""
    rss = hwm = -1
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1])
                if rss >= 0 and hwm >= 0:
                    break
    except Exception:
        return -1, -1
    return rss, hwm


def cuda_alloc_kb():
    if DEVICE == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.memory_allocated() / 1024)
    return -1


def cuda_reserved_kb():
    if DEVICE == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.memory_reserved() / 1024)
    return -1


def mem(phase, op, layer_idx=-1):
    rank = int(os.environ.get("RANK", -1))
    ts = int(time.time() * 1000)
    rss, hwm = read_rss_hwm_kb()
    print(f"MEM|{rank}|{layer_idx}|{phase}|{op}|{rss}|{ts}|{hwm}", file=sys.stderr, flush=True)
    print(f"GPU|{rank}|{layer_idx}|{phase}|{op}|{cuda_alloc_kb()}|{cuda_reserved_kb()}|{ts}",
          file=sys.stderr, flush=True)


# --------------- Sub-model wrappers for ONNX tracing (verbatim from SHAFT-CPU) ---------------
# Each exists because `from_pytorch` traces a module with a fixed signature, and a bare `BertLayer`
# returns a tuple.

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


class ExtendedMaskWrapper(nn.Module):
    """The mask preprocessing, as its own traced sub-graph (sub-graph 0), via the model's own
    get_extended_attention_mask. Built through the protocol rather than directly as zeros so the
    transcript is identical, which phase B requires. dtype is passed explicitly because the stub has
    no parameters left to derive it from by trace time."""

    def __init__(self, bert, dtype):
        super().__init__()
        # A list, so the stripped BERT is not registered as a submodule and `from_pytorch` does not
        # trace it; only its `get_extended_attention_mask` is wanted.
        self._bert = [bert]
        self._dtype = dtype

    def forward(self, attention_mask):
        return self._bert[0].get_extended_attention_mask(
            attention_mask, attention_mask.shape, dtype=self._dtype)


class ClassifierHeadWrapper(nn.Module):
    def __init__(self, pooler, classifier):
        super().__init__()
        self.pooler = pooler
        self.classifier = classifier

    def forward(self, hidden_states):
        pooled = self.pooler(hidden_states)
        return self.classifier(pooled)


def _drop_pytorch_model(g):
    """Drop `from_pytorch`'s reference to the traced module, else the plaintext sub-model stays alive
    as long as its encrypted counterpart."""
    if hasattr(g, "pytorch_model"):
        del g.pytorch_model


def _comm_report(tag, b, r):
    """Per-sub-graph transcript, so a missing round can be localised rather than reasoned about."""
    print(f"SUBCOMM|{tag}|bytes={b}|rounds={r}", flush=True)


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
    rank = ct.comm.get().get_rank()

    # DETERMINISTIC SEEDING, verbatim from `bert_instrumented_gpu.py`. manual_seed() refuses
    # outside debug mode, so debug is enabled only for that call.
    from crypten.config import cfg as _cfg
    _dbg = _cfg.debug.debug_mode
    _cfg.debug.debug_mode = True
    ct.manual_seed(0xDEADBEEF + rank, 0xC0FFEE + rank, 0x5EED)
    _cfg.debug.debug_mode = _dbg
    torch.manual_seed(0x5EED)
    print(f"SEEDED|party={rank}|next=0x{0xDEADBEEF + rank:x}|"
          f"local=0x{0xC0FFEE + rank:x}|global=0x5eed|torch=0x5eed",
          file=sys.stderr, flush=True)

    use_cuda = DEVICE == "cuda" and torch.cuda.is_available()
    if DEVICE == "cuda" and not use_cuda:
        # A silent fall back to CPU would publish a host number under a VRAM step and read as a
        # spectacular reduction. It is a hard stop.
        print(f"FATAL|party={rank}|torch.cuda.is_available()==False, refusing to measure a "
              f"CPU run under a GPU step", file=sys.stderr, flush=True)
        sys.exit(3)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        print(f"GPUINFO|party={rank}|device={DEVICE}|name={torch.cuda.get_device_name(0)}",
              file=sys.stderr, flush=True)

    pid = os.getpid()
    results_dir = os.environ.get("RESULTS_DIR", "/results/shaft")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"party{rank}.pid"), "w") as f:
        f.write(str(pid))

    model_name = os.environ.get("SHAFT_MODEL", "andeskyl/bert-base-cased-sst2")
    max_length = int(os.environ.get("SHAFT_MAX_LENGTH", "128"))
    n_layers = int(os.environ.get("SHAFT_N_LAYERS", "12"))
    print(f"CONFIG|party={rank}|n_layers={n_layers}|seq_len={max_length}|model={model_name}|"
          f"device={DEVICE}", file=sys.stderr, flush=True)

    mem("main", "after_init")

    # ---- Load plaintext model ----
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _ = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        actual_layers = len(model.bert.encoder.layer)
        if n_layers < actual_layers:
            model.bert.encoder.layer = model.bert.encoder.layer[:n_layers]

    mem("main", "after_model_load")

    # ---- Take the sub-models apart and drop the container ----
    # The list entries are replaced with placeholders so `del model` frees everything that is not
    # one of the pieces below, rather than leaving the whole graph alive through one reference.
    embeddings = model.bert.embeddings
    encoder_layers = list(model.bert.encoder.layer)
    pooler = model.bert.pooler
    classifier = model.classifier
    # Kept for the mask sub-graph only; after the four lines below it holds nothing but its config.
    bert_stub = model.bert
    model_dtype = model.dtype

    model.bert.embeddings = nn.Identity()
    model.bert.encoder.layer = nn.ModuleList()
    model.bert.pooler = nn.Identity()
    model.classifier = nn.Identity()
    model.bert = nn.Identity()
    del model
    gc.collect()
    mem("main", "after_extract_submodules")

    # ---- Encrypt inputs ----
    # Same workload as the baseline: the same vocabulary bound, the same length, the same all-ones
    # attention mask (here already in its extended, all-zeros form; see the header).
    input_ids = torch.randint(0, 28996, (1, max_length))
    token_type_ids = torch.zeros(1, max_length, dtype=torch.long)

    attention_mask = torch.ones(1, max_length, dtype=torch.long)
    input_ids_enc = ct.cryptensor(input_ids).to(DEVICE)
    token_type_enc = ct.cryptensor(token_type_ids).to(DEVICE)
    attention_mask_enc = ct.cryptensor(attention_mask).to(DEVICE)

    del input_ids, token_type_ids, attention_mask
    gc.collect()
    mem("main", "after_input_encrypt")

    comm_bytes, comm_rounds = 0, 0
    t0 = time.time()

    # ==== Phase 0: the extended mask, through the model's own code path ====
    # Fifteen sub-graphs rather than fourteen. See ExtendedMaskWrapper for why this is not built
    # directly as zeros: the value would be identical and the TRANSCRIPT would not.
    mask_wrapper = ExtendedMaskWrapper(bert_stub, model_dtype)
    mask_wrapper.eval()
    dummy_mask_in = torch.ones(1, max_length, dtype=torch.long)
    enc_mask_g = ct.nn.from_pytorch(mask_wrapper, (dummy_mask_in,))
    del mask_wrapper, bert_stub, dummy_mask_in
    _drop_pytorch_model(enc_mask_g)
    gc.collect()
    enc_mask_g.encrypt()
    if use_cuda:
        enc_mask_g.cuda()
    with ct.no_grad():
        ext_mask_enc = enc_mask_g(attention_mask_enc)
    b, r = _comm_of(enc_mask_g)
    _comm_report("mask", b, r)
    comm_bytes += b or 0
    comm_rounds += r or 0
    del enc_mask_g, attention_mask_enc
    gc.collect()
    mem("main", "after_extended_mask")

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

    if use_cuda:
        enc_embed.cuda()
        mem("embed", "after_cuda")

    with ct.no_grad():
        hidden = enc_embed(input_ids_enc, token_type_enc)

    b, r = _comm_of(enc_embed)
    _comm_report("embed", b, r)
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
        if use_cuda:
            enc_layer.cuda()
        mem("layer", f"before_run_{i}", i)

        with ct.no_grad():
            hidden = enc_layer(hidden, ext_mask_enc)

        b, r = _comm_of(enc_layer)
        _comm_report(f"layer{i}", b, r)
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
    if use_cuda:
        enc_head.cuda()

    with ct.no_grad():
        logits = enc_head(hidden)

    b, r = _comm_of(enc_head)
    _comm_report("head", b, r)
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

    # Transcript summed over the sub-graphs, printed as integers; phase B asks whether it EQUALS
    # the baseline's single-graph transcript.
    print(f"TRANSCRIPT|party={rank}|bytes={comm_bytes}|rounds={comm_rounds}|"
          f"subgraphs={n_layers + 3}", flush=True)

    if use_cuda:
        peak_alloc = int(torch.cuda.max_memory_allocated() / 1024)
        peak_res = int(torch.cuda.max_memory_reserved() / 1024)
        print(f"GPUPEAK|party={rank}|peak_alloc_kb={peak_alloc}|"
              f"peak_reserved_kb={peak_res}", file=sys.stderr, flush=True)

    if rank == 0:
        predictions = outputs.argmax(dim=-1)
        print(f"Prediction: {predictions.item()}", flush=True)
        # Gate observable (as the baseline): the decrypted output tensor at full precision.
        gate = " ".join(f"{v:.10e}" for v in outputs.flatten().tolist())
        print(f"GATE|logits|{gate}", flush=True)
        print(f"Inference time: {t1 - t0:.2f}s", flush=True)
        if use_cuda:
            print(f"Peak VRAM (party0): allocated={peak_alloc/1024:.1f} MB, "
                  f"reserved={peak_res/1024:.1f} MB", flush=True)

    mem("main", "after_finalize")


if __name__ == "__main__":
    from multiprocess_launcher import MultiProcessLauncher

    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()
