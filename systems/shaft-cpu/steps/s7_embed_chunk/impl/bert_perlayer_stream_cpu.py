#!/usr/bin/env python3
"""
SHAFT-CPU, per-layer variant that STREAMS the checkpoint instead of loading it.

This is `bert_perlayer_disk_cpu.py` with the model load replaced. Its header applies except for the
load; what follows first is why the load changed.

THE OBJECT, from the owner census at `x2_per_layer_disk`'s own peak (`p_census_peak_x2`, census at
rss 1,121,180 kB against a 1,150,240 peak, so 97.5% of it): 866.5 MB of float32 plaintext weights,
in PAIRS with identical counts -- 49 flat against 49 shaped, 1 flat against 1 shaped -- because
`from_pretrained` holds the loaded state_dict AND the instantiated module at once. And the markers
say the peak IS that load:

    after_init                 249,388 kB
    after_model_load         1,150,172        <- the peak, 0.1 s later
    after_extract_submodules   384,688        after the layers reach disk

Every lever the arm carries acts after that instant.

WHAT WAS TRIED FIRST AND FAILED. `low_cpu_mem_load_cpu` passes `low_cpu_mem_usage=True`, which
builds on the meta device and fills tensor by tensor. It removed the duplicate -- `after_model_load`
fell from 1,150,172 to 300,084 kB -- and the run got WORSE, 1,506,944 against 1,139,484, because the
peak moved into the conversion phase 4.5 s later: the mmap-filled weights stay resident as
file-backed pages and add to what conversion needs. Filed as `off-path-runs/x3_low_cpu_mem_load/`.

THIS STEP therefore does not just avoid the duplicate, it never builds the whole model at all.
`safe_open` gives tensor-wise mmap access to the checkpoint, so each scope is pulled on its own, the
encoder layers go straight to disk as state_dicts, and `posix_fadvise(DONTNEED)` drops the pages
that were faulted in. The full plaintext model never exists in this process.

DONOR: the retired campaign's `s6_stream_load`, the last rung of its canonical SHAFT ladder
(962,480 -> 643,784 kB). Its own comment names the object: "Bypass `from_pretrained` to avoid the
state_dict + module duplicate-buffer peak".

WHERE THIS DEPARTS FROM THE DONOR, and it is the reason the number is comparable. That variant reads
per-scope shard files produced by `split_safetensors.py`, which runs OUTSIDE the measured process.
Splitting is not the mechanism -- `safe_open` addresses tensors individually inside ONE file just as
well -- so the shards are dropped and the single `model.safetensors` is read directly. Nothing is
prepared ahead of the run, and `posix_fadvise` is applied to that one file after each scope.

ORIGINAL HEADER OF THE DISK VARIANT FOLLOWS.

SHAFT-CPU, per-layer variant with the plaintext layers held on DISK.

This is `bert_perlayer_cpu.py` plus the retired campaign's `03_disk_offload`, which is the rung
after `02_per_layer` in its ladder (1,834 MB -> 1,322 MB there). Two changes, both mechanical:

  1. Each encoder layer's `state_dict()` is written to disk at startup and the whole plaintext
     model is dropped. Each layer is rebuilt from its file only when its turn comes, and the file
     is removed as soon as it has been read. Without this all twelve plaintext layers sit in RAM
     from load until last use, which on this model is about 469 MB of float32 the protocol never
     touches after conversion.
  2. `malloc_trim(0)` after each `gc.collect()`. This line has measured twice today that a release
     glibc keeps in its arena is invisible to a resident-memory metric, so a per-layer free without
     a trim would free nothing the instrument can see. `embed_chunk_cpu`, already published as
     `s5_embed_chunk`, carries the same call for the same reason.

WHAT IS DELIBERATELY NOT TAKEN from the donor: its `MALLOC_ARENA_MAX=2`. That is a global allocator
knob, and this campaign refused one on 17.08. after the object table showed `[heap]` falling
1,044 MB while `anon` rose 788 MB -- the allocator rebooking between its own buckets with no named
object changing. Leaving it out also keeps this step's gain attributable to the disk offload alone.

The rest of this file is `bert_perlayer_cpu.py` unchanged; its header applies here too.

ORIGINAL HEADER FOLLOWS.

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

    # ---- Stream the checkpoint; the whole model is never built ----
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.bert.modeling_bert import (
        BertEmbeddings, BertLayer, BertPooler,
    )
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    ckpt = hf_hub_download(repo_id=model_name, filename="model.safetensors")

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
    # `BertLayer` is constructed here at all.
    tmpdir = tempfile.mkdtemp(prefix="shaft_layers_", dir=os.environ.get("SNNI_SPILL_DIR", "/tmp"))
    layer_bytes = 0
    for i in range(n_layers):
        st = _scope(f"bert.encoder.layer.{i}.")
        if not st:
            sys.exit(f"FAILED: no tensors under bert.encoder.layer.{i}. in {ckpt}")
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
        embeddings = BertEmbeddings(config)
        pooler = BertPooler(config)
        classifier = nn.Linear(config.hidden_size, config.num_labels)
    embeddings.load_state_dict(_scope("bert.embeddings."))
    pooler.load_state_dict(_scope("bert.pooler."))
    classifier.load_state_dict(_scope("classifier."))
    for _m in (embeddings, pooler, classifier):
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

    from transformers.models.bert.modeling_bert import BertLayer as _BertLayer

    for i in range(n_layers):
        # REBUILT FROM ITS FILE, and the file is removed the moment it has been read.
        #
        # UNDER `fork_rng`, and that is not cosmetic. `BertLayer(config)` initialises random
        # weights before `load_state_dict` overwrites them, and those draws come from torch's
        # GLOBAL generator -- so twelve constructions shift every draw that follows. Measured:
        # without this the arm's logits moved by 5e-4 relative against `x1_per_layer` for a lever
        # that only changes where bytes wait. The weights are overwritten either way, so the draws
        # are pure waste; forking the generator makes them invisible and the gate exact.
        with torch.random.fork_rng():
            layer = _BertLayer(config)
        p = os.path.join(tmpdir, f"layer_{i}.pt")
        layer.load_state_dict(torch.load(p, map_location="cpu"))
        layer.eval()
        os.remove(p)

        wrapper = EncoderLayerWrapper(layer)
        wrapper.eval()
        enc_layer = ct.nn.from_pytorch(wrapper, (dummy_hidden, dummy_mask))
        del wrapper, layer
        _drop_pytorch_model(enc_layer)
        gc.collect()

        enc_layer.encrypt()
        gc.collect()
        malloc_trim()
        mem("layer", f"before_run_{i}", i)

        with ct.no_grad():
            hidden = enc_layer(hidden, ext_mask_enc)

        b, r = _comm_of(enc_layer)
        comm_bytes += b or 0
        comm_rounds += r or 0
        del enc_layer
        gc.collect()
        malloc_trim()
        mem("layer", f"after_run_{i}", i)

    del ext_mask_enc, dummy_hidden, dummy_mask
    os.rmdir(tmpdir)
    gc.collect()
    malloc_trim()

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
