#!/usr/bin/env python3
"""SHAFT-GPU ViT transfer endpoint: ViT-Base/16, one sub-model at a time. Port of bert_perlayer_gpu.py
(s5_per_layer's driver). Differences: no attention mask (no phase 0, fourteen sub-graphs; transcript
compared with the ViT baseline, not the BERT endpoint); no pooler (layernorm, then classifier on the
CLS token); Conv2d patch embedding on pixel_values; checkpoint from a local /vit. Lever set,
per-sub-model construction, markers, transcript and gate unchanged."""

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


# --------------- Sub-model wrappers for ONNX tracing (ported from SHAFT-CPU) ---------------
# Each exists because `from_pytorch` traces a module with a fixed signature.

class EmbeddingWrapper(nn.Module):
    """ViT's embedding is a Conv2d patch projection plus a CLS token and a position table."""

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
        outputs = self.layer(hidden_states)
        return outputs[0]


class ClassifierHeadWrapper(nn.Module):
    """ViT's head: the final LayerNorm, then the classifier on the CLS token (slice [:, 0, :], as
    ViTForImageClassification.forward does). Written out because the model container is dropped by
    this point; the equivalence test checks the two lines against the model's own output."""

    def __init__(self, layernorm, classifier):
        super().__init__()
        self.layernorm = layernorm
        self.classifier = classifier

    def forward(self, hidden_states):
        return self.classifier(self.layernorm(hidden_states)[:, 0, :])


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

    model_name = os.environ.get("SHAFT_MODEL", "/vit")
    # Derived from the checkpoint below and checked against this value; see the baseline driver.
    max_length = int(os.environ.get("SHAFT_MAX_LENGTH", "197"))
    n_layers = int(os.environ.get("SHAFT_N_LAYERS", "12"))
    print(f"CONFIG|party={rank}|n_layers={n_layers}|seq_len={max_length}|model={model_name}|"
          f"device={DEVICE}", file=sys.stderr, flush=True)

    mem("main", "after_init")

    # ---- Load plaintext model ----
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()

    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        actual_layers = len(model.vit.encoder.layer)
        if n_layers < actual_layers:
            model.vit.encoder.layer = model.vit.encoder.layer[:n_layers]

    cfg = model.config
    _img = cfg.image_size
    _hidden = cfg.hidden_size
    _tok = (cfg.image_size // cfg.patch_size) ** 2 + 1
    print(f"VIT|image={cfg.image_size}|patch={cfg.patch_size}|tokens={_tok}|hidden={_hidden}"
          f"|heads={cfg.num_attention_heads}|ffn={cfg.intermediate_size}",
          file=sys.stderr, flush=True)
    if _tok != max_length:
        sys.exit(f"FAILED: checkpoint has {_tok} tokens, SHAFT_MAX_LENGTH says {max_length}")

    mem("main", "after_model_load")

    # ---- Take the sub-models apart and drop the container ----
    # List entries replaced with placeholders so `del model` frees everything not one of the pieces
    # below. No bert_stub/model_dtype: those traced the extended mask, and ViT has no mask.
    embeddings = model.vit.embeddings
    encoder_layers = list(model.vit.encoder.layer)
    layernorm = model.vit.layernorm
    classifier = model.classifier

    model.vit.embeddings = nn.Identity()
    model.vit.encoder.layer = nn.ModuleList()
    model.vit.layernorm = nn.Identity()
    model.classifier = nn.Identity()
    model.vit = nn.Identity()
    del model
    gc.collect()
    mem("main", "after_extract_submodules")

    # ---- Encrypt inputs ----
    # Same workload as the baseline: one pseudo-image from the pinned generator, no mask.
    pixel_values = torch.randn(1, cfg.num_channels, _img, _img)
    pixel_enc = ct.cryptensor(pixel_values).to(DEVICE)

    del pixel_values
    gc.collect()
    mem("main", "after_input_encrypt")

    # Sub-graphs are counted, not computed from a constant: the first version inherited the BERT
    # driver's n_layers+3 (mask+embedding+head) and printed subgraphs=15 for a fourteen-piece ViT
    # decomposition. Nothing measured was affected, but the label was a false structural claim.
    comm_bytes, comm_rounds, n_subgraphs = 0, 0, 0
    t0 = time.time()

    # ==== Phase 0 DOES NOT EXIST HERE ====
    # ViT has no attention mask, so the BERT driver's extended-mask sub-graph is absent: fourteen
    # sub-graphs follow, not fifteen.

    # ==== Phase 1: embeddings ====
    embed_wrapper = EmbeddingWrapper(embeddings)
    embed_wrapper.eval()
    dummy_pix = torch.zeros(1, cfg.num_channels, _img, _img)

    enc_embed = ct.nn.from_pytorch(embed_wrapper, (dummy_pix,))
    del embed_wrapper, embeddings, dummy_pix
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
        hidden = enc_embed(pixel_enc)

    b, r = _comm_of(enc_embed)
    _comm_report("embed", b, r)
    comm_bytes += b or 0
    comm_rounds += r or 0
    n_subgraphs += 1
    del enc_embed, pixel_enc
    gc.collect()
    mem("embed", "after_run")

    # ==== Phase 2: encoder layers, one at a time ====
    dummy_hidden = torch.randn(1, max_length, _hidden)

    for i in range(n_layers):
        layer = encoder_layers[i]
        encoder_layers[i] = None          # the list must not keep it alive

        wrapper = EncoderLayerWrapper(layer)
        wrapper.eval()
        enc_layer = ct.nn.from_pytorch(wrapper, (dummy_hidden,))
        del wrapper, layer
        _drop_pytorch_model(enc_layer)
        gc.collect()

        enc_layer.encrypt()
        gc.collect()
        if use_cuda:
            enc_layer.cuda()
        mem("layer", f"before_run_{i}", i)

        with ct.no_grad():
            hidden = enc_layer(hidden)

        b, r = _comm_of(enc_layer)
        _comm_report(f"layer{i}", b, r)
        comm_bytes += b or 0
        comm_rounds += r or 0
        n_subgraphs += 1
        del enc_layer
        gc.collect()
        mem("layer", f"after_run_{i}", i)

    del dummy_hidden, encoder_layers
    gc.collect()

    # ==== Phase 3: the final LayerNorm and the classifier on the CLS token ====
    head_wrapper = ClassifierHeadWrapper(layernorm, classifier)
    head_wrapper.eval()
    dummy_head = torch.randn(1, max_length, _hidden)

    enc_head = ct.nn.from_pytorch(head_wrapper, dummy_head)
    del head_wrapper, layernorm, classifier, dummy_head
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
    n_subgraphs += 1
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
    # the baseline's single-graph transcript. Hard stop if the accumulated sub-graph count is wrong,
    # since a decomposition that traced a different number of pieces than it thinks is a plausible artefact.
    _expect = n_layers + 2          # embedding + n layers + head, and NO mask sub-graph on ViT
    if n_subgraphs != _expect:
        sys.exit(f"FAILED: accumulated {n_subgraphs} sub-graphs, expected {_expect}")
    print(f"TRANSCRIPT|party={rank}|bytes={comm_bytes}|rounds={comm_rounds}|"
          f"subgraphs={n_subgraphs}", flush=True)

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
