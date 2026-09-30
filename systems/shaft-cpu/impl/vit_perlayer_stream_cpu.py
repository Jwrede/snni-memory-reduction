#!/usr/bin/env python3
"""SHAFT-CPU ViT transfer endpoint: ViT-Base/16 per layer, streamed from the checkpoint.
Port of bert_perlayer_stream_cpu.py (s8_encrypt_chunk's driver). Model-specific: no attention mask in
ViTLayer; head = final layernorm + classifier on CLS (no pooler); Conv2d patch embedding over
pixel_values (scope vit.embeddings.); local checkpoint directory. Identical: seven levers, streaming,
per-layer construction under fork_rng, markers, transcript, gate. Marker format unchanged.
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


# Sub-model wrappers for ONNX tracing (from 02_per_layer): from_pytorch traces a fixed signature,
# and a bare ViTLayer returns a tuple.

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
    """ViT's head: final LayerNorm, then classifier on the CLS token (sequence_output[:,0,:] after
    the layernorm), reproducing ViTForImageClassification.forward rather than approximating it."""

    def __init__(self, layernorm, classifier):
        super().__init__()
        self.layernorm = layernorm
        self.classifier = classifier

    def forward(self, hidden_states):
        return self.classifier(self.layernorm(hidden_states)[:, 0, :])


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

    # ---- Stream the checkpoint; the whole model is never built ----
    from transformers import AutoConfig
    from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer
    from safetensors import safe_open

    # No tokenizer and no hub: the checkpoint is a local directory bound into the container (nodes
    # have no network). Setup, identical for baseline and endpoint.
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

    # Encoder layers go straight from file to disk: each scope faulted in, written as a state_dict
    # and dropped, so at most one layer's weights are resident and no `ViTLayer` is built here.
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

    # The three sub-models not streamed per use are built now, each from its own scope.
    with torch.random.fork_rng():
        embeddings = ViTEmbeddings(config)
        layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        classifier = nn.Linear(config.hidden_size, config.num_labels)
    # strict=False on the embeddings only: ViTEmbeddings carries a mask_token parameter (masked-image
    # pretraining) absent from this classification checkpoint. Everything else is still checked, so a
    # missing WEIGHT (as opposed to that one buffer) still fails.
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

    # Nothing to take apart (sub-models built above, layers already on disk); marker kept so the
    # phase series stays comparable with the arm.
    mem("main", "after_extract_submodules")

    # ---- Encrypt inputs ----
    # One encrypted operand, no mask: ViT has no padding, so no extended mask to build, encrypt or
    # carry through twelve layers. The baseline driver is identical in this.
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
        # Rebuilt from its file (removed on read) under fork_rng: ViTLayer(config)'s random init
        # draws from torch's GLOBAL generator, so 12 constructions would shift every later draw
        # (measured 5e-4 logit shift vs x1_per_layer). Weights are overwritten anyway; forking keeps
        # the gate exact.
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
