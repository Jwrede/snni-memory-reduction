#!/usr/bin/env python3
"""SHAFT GPU s2_per_layer (cumulative on s1): decompose the model and move each encrypted sub-module
to CUDA just before its run, freeing it after, so only ONE layer's weights sit in VRAM at a time.
Instrumentation and this file host all later per-layer levers, gated by their SHAFT_* env vars."""

import gc
import os
import sys
import time
import threading
import contextlib
import ctypes
import torch
import torch.nn as nn
import crypten
import graphop_markers_gpu  # per-op MEM+GPU markers (CPU-consistent)
import embed_chunk_gpu      # s1 lever (no-op unless SHAFT_EMBED_CHUNKS>1)
import matmul_chunk_gpu     # s5 lever: chunk the FFN matmul (no-op unless SHAFT_MATMUL_CHUNKS>1)
import matmul_limb_loop_gpu  # s7 lever: loop CUDALongTensor limb-products (no-op unless SHAFT_LIMB_LOOP=1)
import crypten as ct
import crypten.nn.module as ctm
from crypten import cfg

DEVICE = os.environ.get("SHAFT_DEVICE", "cuda")

# s3 lever (no-op unless SHAFT_POLY_GELU=1): poly GELU + fewer softmax ODE iters (mirrors CPU s4_reduce).
_POLY = os.environ.get("SHAFT_POLY_GELU", "0") == "1"

# s4 lever (no-op unless SHAFT_MALLOC_TRIM=1): return freed host pages to the OS after each sub-module.
_MALLOC_TRIM = os.environ.get("SHAFT_MALLOC_TRIM", "0") == "1"

# s8 lever (SHAFT_STREAM_LOAD): per-scope safetensors stream-load, one shard at a time with page
# cache dropped, so file-backed pages cannot accumulate in VmRSS across the 12 layers.
_STREAM_LOAD = os.environ.get("SHAFT_STREAM_LOAD", "0") == "1"

# s13 lever (SHAFT_EMBED_STREAM): move the word-embedding table back to host, streamed per chunk;
# targets the embedding phase, which is the whole VRAM wall once layers are down to 36-92 MiB.
_EMBED_STREAM = os.environ.get("SHAFT_EMBED_STREAM", "0") == "1"

# s9 lever (SHAFT_TRACE_ONCE): trace+encrypt the encoder-layer graph once and reuse it, swapping only
# the encrypted weights per layer, avoiding 11 of the 12 from_pytorch ONNX traces.
_TRACE_ONCE = os.environ.get("SHAFT_TRACE_ONCE", "0") == "1"

# s11 lever (needs TRACE_ONCE + STREAM_LOAD): load each layer's weights directly to the device and
# build cryptensors there, so neither plaintext nor int64 shares materialize in host RAM.
_DEV_ENCRYPT = os.environ.get("SHAFT_DEV_ENCRYPT", "0") == "1"
_SEED = int(os.environ.get("SHAFT_SEED", "0"))
try:
    _LIBC = ctypes.CDLL("libc.so.6")
except Exception:
    _LIBC = None


def _fn_ctx():
    if _POLY:
        return cfg.temp_override({
            "functions.gelu_method": "poly",
            "functions.softmax_ode_iter_num": 8,
        })
    return contextlib.nullcontext()


def read_vmrss_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        return -1
    return -1


def read_vmpeak_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmPeak:"):
                    return int(line.split()[1])
    except Exception:
        return -1
    return -1


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
    print(f"MEM|{rank}|{layer_idx}|{phase}|{op}|{read_vmrss_kb()}|{ts}",
          file=sys.stderr, flush=True)
    print(f"GPU|{rank}|{layer_idx}|{phase}|{op}|{cuda_alloc_kb()}|{cuda_reserved_kb()}|{ts}",
          file=sys.stderr, flush=True)


def start_vmrss_poller(rank, interval_ms=100, output_dir="/results/shaft"):
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"party{rank}_vmrss.log")
    pid = os.getpid()
    sleep_sec = interval_ms / 1000.0

    def poller():
        with open(outfile, "w") as f:
            f.write(f"# Polling PID={pid} (party {rank}) every {interval_ms}ms\n")
            f.write("# timestamp_ms VmRSS_kB VmPeak_kB cuda_alloc_kB cuda_reserved_kB\n")
            while True:
                ts = int(time.time() * 1000)
                f.write(f"{ts} {read_vmrss_kb()} {read_vmpeak_kb()} "
                        f"{cuda_alloc_kb()} {cuda_reserved_kb()}\n")
                f.flush()
                time.sleep(sleep_sec)

    t = threading.Thread(target=poller, daemon=True)
    t.start()
    return t


def _free_device():
    gc.collect()
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _MALLOC_TRIM and _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


# --------------- s8 stream-load helpers (ported from CPU 08_low_cpu_mem_load) ---------------

def _drop_file_cache(path):
    """Drop a file's page-cache pages so its mmap'd bytes leave VmRSS."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _load_shard_then_drop(shards_dir, scope):
    """Load ONE scope's safetensors shard, then drop its page cache."""
    from safetensors import safe_open
    path = os.path.join(shards_dir, f"{scope}.safetensors")
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    _drop_file_cache(path)
    return out


def _stream_load_setup(model_name, n_layers):
    """Build the sub-modules WITHOUT from_pretrained: one scope shard at a time.
    Returns (config, embeddings, pooler, classifier, tmpdir) with each layer's state_dict on disk."""
    import tempfile
    from transformers import AutoConfig
    from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler
    from huggingface_hub import hf_hub_download

    config = AutoConfig.from_pretrained(model_name)
    snapshot_dir = os.path.dirname(hf_hub_download(repo_id=model_name, filename="model.safetensors"))
    shards_dir = os.path.join(snapshot_dir, "shards")
    if not os.path.isdir(shards_dir):
        import subprocess
        subprocess.run([sys.executable, "/root/shaft/split_safetensors.py", snapshot_dir], check=True)

    tmpdir = tempfile.mkdtemp(prefix="shaft_layers_", dir=os.environ.get("SHAFT_TMPDIR", "/tmp"))
    for i in range(n_layers):
        st = _load_shard_then_drop(shards_dir, f"bert.encoder.layer.{i}")
        torch.save(st, os.path.join(tmpdir, f"layer_{i}.pt"))
        del st
        _free_device()

    embeddings = BertEmbeddings(config)
    embeddings.load_state_dict(_load_shard_then_drop(shards_dir, "bert.embeddings"))
    embeddings.eval()
    pooler = BertPooler(config)
    pooler.load_state_dict(_load_shard_then_drop(shards_dir, "bert.pooler"))
    pooler.eval()
    classifier = nn.Linear(config.hidden_size, config.num_labels)
    classifier.load_state_dict(_load_shard_then_drop(shards_dir, "classifier"))
    classifier.eval()
    _free_device()
    return config, embeddings, pooler, classifier, tmpdir


# --------------- Sub-module wrappers (identical to CPU 02_per_layer) ---------------

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


def main():
    ct.init()
    rank = ct.comm.get().get_rank()

    use_cuda = DEVICE == "cuda" and torch.cuda.is_available()
    if DEVICE == "cuda" and not use_cuda:
        print(f"FATAL|party={rank}|torch.cuda.is_available()==False", file=sys.stderr, flush=True)
        sys.exit(2)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        print(f"GPUINFO|party={rank}|device={DEVICE}|name={torch.cuda.get_device_name(0)}|"
              f"visible={os.environ.get('CUDA_VISIBLE_DEVICES','?')}", file=sys.stderr, flush=True)

    results_dir = os.environ.get("RESULTS_DIR", "/results/shaft")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"party{rank}.pid"), "w") as f:
        f.write(str(os.getpid()))

    start_vmrss_poller(rank, interval_ms=100, output_dir=results_dir)

    model_name = os.environ.get("SHAFT_MODEL", "andeskyl/bert-base-cased-sst2")
    max_length = int(os.environ.get("SHAFT_MAX_LENGTH", "128"))
    n_layers = int(os.environ.get("SHAFT_N_LAYERS", "12"))
    print(f"CONFIG|party={rank}|n_layers={n_layers}|seq_len={max_length}|model={model_name}|"
          f"device={DEVICE}|per_layer=1|embed_chunks={os.environ.get('SHAFT_EMBED_CHUNKS','1')}|"
          f"poly_gelu={int(_POLY)}|malloc_trim={int(_MALLOC_TRIM)}|stream_load={int(_STREAM_LOAD)}"
          f"|trace_once={int(_TRACE_ONCE)}|dev_encrypt={int(_DEV_ENCRYPT)}|embed_stream={int(_EMBED_STREAM)}|seed={_SEED}",
          file=sys.stderr, flush=True)

    if _SEED:
        torch.manual_seed(_SEED)
    mem("main", "after_init")

    config = None
    tmpdir = None
    encoder_layers = None
    if _STREAM_LOAD:
        # s8: never materialize the whole model; one scope shard at a time, page cache dropped after each.
        config, embeddings, pooler, classifier, tmpdir = _stream_load_setup(model_name, n_layers)
        mem("main", "after_model_load")
        mem("main", "after_extract_submodules")
    else:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
            actual_layers = len(model.bert.encoder.layer)
            if n_layers < actual_layers:
                model.bert.encoder.layer = model.bert.encoder.layer[:n_layers]
        mem("main", "after_model_load")

        # Extract sub-modules, free the full model structure
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

    # Encrypt inputs (on device)
    input_ids = torch.randint(0, 28996, (1, max_length))
    token_type_ids = torch.zeros(1, max_length, dtype=torch.long)
    input_ids_enc = ct.cryptensor(input_ids).to(DEVICE)
    token_type_enc = ct.cryptensor(token_type_ids).to(DEVICE)
    ext_mask_enc = ct.cryptensor(torch.zeros(1, 1, 1, max_length)).to(DEVICE)
    del input_ids, token_type_ids
    gc.collect()
    mem("main", "after_input_encrypt")

    t0 = time.time()

    # ==== Phase 1: Embeddings (streamed to device, freed after) ====
    embed_wrapper = EmbeddingWrapper(embeddings)
    embed_wrapper.eval()
    dummy_ids = torch.zeros(1, max_length, dtype=torch.long)
    dummy_tt = torch.zeros(1, max_length, dtype=torch.long)
    enc_embed = ct.nn.from_pytorch(embed_wrapper, (dummy_ids, dummy_tt))
    del embed_wrapper, embeddings, dummy_ids, dummy_tt
    if hasattr(enc_embed, "pytorch_model"):
        del enc_embed.pytorch_model
    gc.collect()
    enc_embed.encrypt()
    if use_cuda:
        enc_embed.cuda()
        if _EMBED_STREAM:
            # move ONLY the big word-embedding table back to the host (streamed per chunk in the matmul)
            _moved = 0
            for _n, _m in enc_embed._modules.items():
                if isinstance(_m, ctm.Parameter):
                    _d = _m._parameters.get("data", None)
                    if _d is not None and getattr(_d, "dim", lambda: 0)() == 2 and _d.shape[0] > 10000:
                        _m.set_parameter("data", _d.to("cpu"))
                        _moved += 1
            print(f"EMBEDSTREAM|moved_tables={_moved}", file=sys.stderr, flush=True)
    _free_device()
    mem("embed", "after_encrypt")

    with _fn_ctx(), ct.no_grad():
        hidden = enc_embed(input_ids_enc, token_type_enc)
    del enc_embed, input_ids_enc, token_type_enc
    _free_device()
    mem("embed", "after_run")

    # ==== Phase 2: Encoder layers, ONE resident on device at a time ====
    dummy_hidden = torch.randn(1, max_length, 768)
    dummy_mask = torch.zeros(1, 1, 1, max_length)
    _tmpl = None
    for i in range(n_layers):
        if _DEV_ENCRYPT and _TRACE_ONCE and _STREAM_LOAD and _tmpl is not None:
            # s11: weights load straight to the device, cryptensors built there, traced graph reused.
            _lp = os.path.join(tmpdir, f"layer_{i}.pt")
            _raw = torch.load(_lp, map_location=DEVICE)
            os.remove(_lp)
            for _n, _m in _tmpl._modules.items():
                if isinstance(_m, ctm.Parameter):
                    _k = _n[len("layer."):] if _n.startswith("layer.") else _n
                    if _k in _raw:
                        _m.set_parameter("data", ct.cryptensor(_raw[_k], src=0))
                        _raw[_k] = None
            del _raw
            enc_layer = _tmpl
            gc.collect()
            _free_device()
            mem("layer", f"before_run_{i}", i)
            with _fn_ctx(), ct.no_grad():
                hidden = enc_layer(hidden, ext_mask_enc)
            _free_device()
            mem("layer", f"after_run_{i}", i)
            continue
        if _STREAM_LOAD:
            # load THIS layer's weights from its own file, then delete the file
            from transformers.models.bert.modeling_bert import BertLayer
            layer = BertLayer(config)
            _lp = os.path.join(tmpdir, f"layer_{i}.pt")
            layer.load_state_dict(torch.load(_lp, map_location="cpu"))
            layer.eval()
            os.remove(_lp)
        else:
            layer = encoder_layers[i]
            encoder_layers[i] = None
        wrapper = EncoderLayerWrapper(layer)
        wrapper.eval()
        if _TRACE_ONCE and _tmpl is not None:
            # reuse the already traced+encrypted graph: swap ONLY the encrypted weights in place
            _sd = dict(wrapper.state_dict())
            for _n, _m in _tmpl._modules.items():
                if isinstance(_m, ctm.Parameter) and _n in _sd:
                    _m.set_parameter("data", ct.cryptensor(_sd[_n], src=0).to(DEVICE))
            del _sd, wrapper, layer
            enc_layer = _tmpl
            gc.collect()
        else:
            enc_layer = ct.nn.from_pytorch(wrapper, (dummy_hidden, dummy_mask))
            del wrapper, layer
            if hasattr(enc_layer, "pytorch_model"):
                del enc_layer.pytorch_model
            gc.collect()
            enc_layer.encrypt()
            if use_cuda:
                enc_layer.cuda()      # <- only this layer's weights on the device
            if _TRACE_ONCE:
                _tmpl = enc_layer
        _free_device()
        mem("layer", f"before_run_{i}", i)

        with _fn_ctx(), ct.no_grad():
            hidden = enc_layer(hidden, ext_mask_enc)

        if not _TRACE_ONCE:
            del enc_layer             # <- free this layer's device weights before the next
        _free_device()
        mem("layer", f"after_run_{i}", i)

    if _TRACE_ONCE and _tmpl is not None:
        del _tmpl
    del ext_mask_enc, dummy_hidden, dummy_mask, encoder_layers
    _free_device()

    # ==== Phase 3: Pooler + Classifier ====
    head_wrapper = ClassifierHeadWrapper(pooler, classifier)
    head_wrapper.eval()
    dummy_head = torch.randn(1, max_length, 768)
    enc_head = ct.nn.from_pytorch(head_wrapper, dummy_head)
    del head_wrapper, pooler, classifier, dummy_head
    if hasattr(enc_head, "pytorch_model"):
        del enc_head.pytorch_model
    gc.collect()
    enc_head.encrypt()
    if use_cuda:
        enc_head.cuda()
    _free_device()
    with _fn_ctx(), ct.no_grad():
        logits = enc_head(hidden)
    del enc_head, hidden
    _free_device()
    t1 = time.time()
    mem("head", "after_run")

    outputs = logits.get_plain_text()
    del logits
    _free_device()
    mem("main", "after_decrypt")

    if use_cuda:
        peak_alloc = int(torch.cuda.max_memory_allocated() / 1024)
        peak_res = int(torch.cuda.max_memory_reserved() / 1024)
        print(f"GPUPEAK|party={rank}|peak_alloc_kb={peak_alloc}|peak_reserved_kb={peak_res}",
              file=sys.stderr, flush=True)

    if rank == 0:
        print(f"LOGITS|{[round(float(x), 6) for x in outputs.flatten().tolist()]}", flush=True)
        predictions = outputs.argmax(dim=-1)
        print(f"Prediction: {predictions.item()}", flush=True)
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
