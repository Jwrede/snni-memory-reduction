#!/usr/bin/env python3
"""
SHAFT-CPU ViT transfer, ENDPOINT: ViT-Base/16 per layer, streamed from the checkpoint.

The port of `bert_perlayer_stream_cpu.py`, which is `s8_encrypt_chunk`'s driver, to the one model
in this campaign that is a genuinely different architecture. `GEOMETRY-TRANSFER.md` says why this
is the only system where that sentence means anything.

WHAT THE ARCHITECTURE FORCES, and it is less than one might expect:

  1. `ViTLayer` takes NO attention mask. BERT's layer signature is
     `layer(hidden_states, attention_mask=...)`; ViT's is `layer(hidden_states, head_mask=None)`.
     So the extended-mask tensor, its encryption and the wrapper argument all disappear. That is
     one encrypted operand fewer per layer, and it is a property of the model rather than a lever.
  2. There is NO POOLER. BERT's head is `pooler` then `classifier`; ViT's is a final `layernorm`
     then `classifier` on the CLS token. So the head wrapper is built from `vit.layernorm.` and
     `classifier.` and slices token 0 itself.
  3. The embedding is a Conv2d patch projection plus a CLS token and a position table, not a
     vocabulary lookup, so its input is `pixel_values` and its scope is `vit.embeddings.`.
  4. The checkpoint is a LOCAL DIRECTORY, because the measured image bakes in only the BERT weights
     and the compute nodes have no network. `hf_hub_download` is therefore replaced by a path join,
     which is setup and is identical for the baseline and this endpoint.

WHAT IS IDENTICAL, and this is the point of the exercise: the seven levers, the streaming, the
per-layer construction under `fork_rng`, the markers, the transcript accumulation and the gate.
Nothing about the REDUCTION is re-derived for this model. If the numbers transfer, they transfer
against an unchanged lever set; if they do not, that is the finding.

Marker format on stderr, unchanged.
"""

import ctypes
import gc
import importlib
import os
import sys
import tempfile
import time

import torch
import torch.nn as nn

import crypten
import crypten as ct

for _lv in filter(None, (m.strip() for m in os.environ.get("SHAFT_LEVERS", "").split(","))):
    importlib.import_module(_lv)


try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def malloc_trim():
    """Hand freed pages back to the kernel. Without it a per-layer free is invisible here."""
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


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
# module with a fixed signature, and a bare `ViTLayer` returns a tuple.

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
        outputs = self.layer(hidden_states)
        return outputs[0]


class ClassifierHeadWrapper(nn.Module):
    """ViT's head: final LayerNorm, then the classifier on the CLS token.

    BERT pools with a learned dense-plus-tanh over token 0; ViT normalises and reads token 0
    directly. `ViTForImageClassification.forward` is `sequence_output[:, 0, :]` after the
    layernorm, and that is what this reproduces, so the decomposition computes the model's own
    function rather than a convenient approximation of it.
    """

    def __init__(self, layernorm, classifier):
        super().__init__()
        self.layernorm = layernorm
        self.classifier = classifier

    def forward(self, hidden_states):
        return self.classifier(self.layernorm(hidden_states)[:, 0, :])


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

    # ---- Stream the checkpoint; the whole model is never built ----
    from transformers import AutoConfig
    from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer
    from safetensors import safe_open

    # NO TOKENIZER and NO HUB. The checkpoint is a local directory bound into the container,
    # because the measured image carries only the BERT weights and the compute nodes have no
    # network. Setup, identical for the baseline and for this endpoint.
    config = AutoConfig.from_pretrained(model_name)
    ckpt = os.path.join(model_name, "model.safetensors")
    if not os.path.exists(ckpt):
        sys.exit(f"FAILED: no model.safetensors under {model_name}")
    max_length = (config.image_size // config.patch_size) ** 2 + 1
    print(f"VIT|image={config.image_size}|patch={config.patch_size}|tokens={max_length}"
          f"|hidden={config.hidden_size}|heads={config.num_attention_heads}"
          f"|ffn={config.intermediate_size}", file=sys.stderr, flush=True)

    def _drop_cache(path):
        """Tell the kernel it may forget the pages this read faulted in."""
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass

    def _scope(prefix):
        """The tensors under `prefix`, keyed relative to it, pulled one at a time."""
        out = {}
        with safe_open(ckpt, framework="pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    out[k[len(prefix):]] = f.get_tensor(k)
        _drop_cache(ckpt)
        return out

    # THE ENCODER LAYERS GO STRAIGHT FROM THE FILE TO DISK. Each scope is faulted in, written out
    # as a state_dict and dropped, so at most one layer's weights are resident at a time and no
    # `ViTLayer` is constructed here at all.
    tmpdir = tempfile.mkdtemp(prefix="shaft_layers_", dir=os.environ.get("SNNI_SPILL_DIR", "/tmp"))
    layer_bytes = 0
    for i in range(n_layers):
        st = _scope(f"vit.encoder.layer.{i}.")
        if not st:
            sys.exit(f"FAILED: no tensors under vit.encoder.layer.{i}. in {ckpt}")
        p = os.path.join(tmpdir, f"layer_{i}.pt")
        torch.save(st, p)
        layer_bytes += os.path.getsize(p)
        del st
        gc.collect()
        malloc_trim()
    print(f"LEVER|stream_load|layers={n_layers}|bytes={layer_bytes}|ckpt={ckpt}",
          file=sys.stderr, flush=True)

    # The three sub-models that are NOT streamed per use are built now, each from its own scope.
    with torch.random.fork_rng():
        embeddings = ViTEmbeddings(config)
        layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        classifier = nn.Linear(config.hidden_size, config.num_labels)
    # `strict=False` on the embeddings ONLY, and the reason is named rather than convenient:
    # `ViTEmbeddings` carries a `mask_token` parameter used by masked-image pretraining, which this
    # classification checkpoint does not contain. Everything else must match, so the call is checked
    # and a missing WEIGHT (as opposed to that one buffer) still fails.
    _emb_missing, _emb_unexpected = embeddings.load_state_dict(_scope("vit.embeddings."), strict=False)
    _emb_missing = [k for k in _emb_missing if k != "mask_token"]
    if _emb_missing or _emb_unexpected:
        sys.exit(f"FAILED: embeddings state_dict mismatch, missing={_emb_missing} "
                 f"unexpected={list(_emb_unexpected)}")
    layernorm.load_state_dict(_scope("vit.layernorm."))
    classifier.load_state_dict(_scope("classifier."))
    for _m in (embeddings, layernorm, classifier):
        _m.eval()
    gc.collect()
    malloc_trim()

    mem("main", "after_model_load")

    # NOTHING TO TAKE APART: the sub-models were built from their own scopes above and the
    # encoder layers are already on disk. The marker is kept so the phase series is comparable
    # with the rest of the arm.
    mem("main", "after_extract_submodules")

    # ---- Encrypt inputs ----
    # Same workload as the baseline: the same vocabulary bound, the same length, the same
    # all-ones attention mask (here already in its extended, all-zeros form; see the header).
    # ONE encrypted operand, and no mask at all: ViT has no padding, so there is no extended mask
    # to build, encrypt or carry through twelve layers. The baseline driver is identical in this,
    # so the two are comparable.
    _img = config.image_size
    pixel_values = torch.randn(1, config.num_channels, _img, _img)
    pixel_enc = ct.cryptensor(pixel_values).to(device)

    del pixel_values
    gc.collect()
    mem("main", "after_input_encrypt")

    comm_bytes, comm_rounds = 0, 0
    t0 = time.time()

    # ==== Phase 1: embeddings ====
    embed_wrapper = EmbeddingWrapper(embeddings)
    embed_wrapper.eval()
    dummy_pix = torch.zeros(1, config.num_channels, _img, _img)

    enc_embed = ct.nn.from_pytorch(embed_wrapper, (dummy_pix,))
    del embed_wrapper, embeddings, dummy_pix
    _drop_pytorch_model(enc_embed)
    gc.collect()
    mem("embed", "after_convert")

    enc_embed.encrypt()
    gc.collect()
    mem("embed", "after_encrypt")

    with ct.no_grad():
        hidden = enc_embed(pixel_enc)

    b, r = _comm_of(enc_embed)
    comm_bytes += b or 0
    comm_rounds += r or 0
    del enc_embed, pixel_enc
    gc.collect()
    mem("embed", "after_run")

    # ==== Phase 2: encoder layers, one at a time ====
    dummy_hidden = torch.randn(1, max_length, config.hidden_size)

    from transformers.models.vit.modeling_vit import ViTLayer as _ViTLayer

    for i in range(n_layers):
        # REBUILT FROM ITS FILE, and the file is removed the moment it has been read.
        #
        # UNDER `fork_rng`, and that is not cosmetic. `ViTLayer(config)` initialises random
        # weights before `load_state_dict` overwrites them, and those draws come from torch's
        # GLOBAL generator -- so twelve constructions shift every draw that follows. Measured:
        # without this the arm's logits moved by 5e-4 relative against `x1_per_layer` for a lever
        # that only changes where bytes wait. The weights are overwritten either way, so the draws
        # are pure waste; forking the generator makes them invisible and the gate exact.
        with torch.random.fork_rng():
            layer = _ViTLayer(config)
        p = os.path.join(tmpdir, f"layer_{i}.pt")
        layer.load_state_dict(torch.load(p, map_location="cpu"))
        layer.eval()
        os.remove(p)

        wrapper = EncoderLayerWrapper(layer)
        wrapper.eval()
        enc_layer = ct.nn.from_pytorch(wrapper, (dummy_hidden,))
        del wrapper, layer
        _drop_pytorch_model(enc_layer)
        gc.collect()

        enc_layer.encrypt()
        gc.collect()
        malloc_trim()
        mem("layer", f"before_run_{i}", i)

        with ct.no_grad():
            hidden = enc_layer(hidden)

        b, r = _comm_of(enc_layer)
        comm_bytes += b or 0
        comm_rounds += r or 0
        del enc_layer
        gc.collect()
        malloc_trim()
        mem("layer", f"after_run_{i}", i)

    del dummy_hidden
    os.rmdir(tmpdir)
    gc.collect()
    malloc_trim()

    # ==== Phase 3: final layernorm and classifier ====
    head_wrapper = ClassifierHeadWrapper(layernorm, classifier)
    head_wrapper.eval()
    dummy_head = torch.randn(1, max_length, config.hidden_size)

    enc_head = ct.nn.from_pytorch(head_wrapper, dummy_head)
    del head_wrapper, layernorm, classifier, dummy_head
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
