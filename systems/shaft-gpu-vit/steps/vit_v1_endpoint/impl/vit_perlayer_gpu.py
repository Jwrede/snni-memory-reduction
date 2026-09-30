#!/usr/bin/env python3
"""
SHAFT-GPU ViT transfer, ENDPOINT: ViT-Base/16 run one sub-model at a time, on an H200.

The port of `bert_perlayer_gpu.py`, which is `s5_per_layer`'s driver, to the one model in this
campaign that is a genuinely different architecture. `GEOMETRY-TRANSFER.md` says why only the two
SHAFT lines can be asked this at all: every other system computes against shape parameters and loads
no model, so for them "ViT" would be an integer.

WHY THIS EXISTS BESIDE SHAFT-CPU's ANSWER. That system's transfer is a HOST-pool result. This line's
reduction is 19.232x on VRAM against 1.365x on host, so the device pool's answer is a different one
and no other two-pool system in the campaign loads a model.

WHAT THE ARCHITECTURE FORCES, and it is less than one might expect:

  1. `ViTLayer` takes NO attention mask. BERT's layer signature is
     `layer(hidden_states, attention_mask=...)`; ViT's is `layer(hidden_states, head_mask=None)`.
     So the extended-mask tensor, its encryption, its whole sub-graph and the wrapper argument all
     disappear. **That removes Phase 0 outright**, so this driver traces FOURTEEN sub-graphs where
     the BERT one traces fifteen, and it is one encrypted operand fewer per layer. It is a property
     of the model, not a lever, and it is the reason this endpoint's transcript is NOT expected to
     equal the BERT endpoint's: the two run different protocols on different models. What phase B
     asks here is that the ViT ENDPOINT matches the ViT BASELINE, and that pair is what
     `vit_v1_endpoint` is judged on.
  2. There is NO POOLER. BERT's head is `pooler` then `classifier`; ViT's is a final `layernorm`
     then `classifier` on the CLS token. So the head wrapper is built from `vit.layernorm` and
     `classifier` and slices token 0 itself, which is what `ViTForImageClassification.forward` does.
  3. The embedding is a Conv2d patch projection plus a CLS token and a position table, not a
     vocabulary lookup, so its input is `pixel_values` and there is no `token_type_ids`.
  4. The checkpoint is a LOCAL DIRECTORY, because the measured image bakes in only the BERT weights
     and the compute nodes have no network. That is setup and is identical for the baseline and this
     endpoint, exactly as the pinned seed is.

WHAT IS IDENTICAL, and this is the point of the exercise: the lever set, the per-sub-model
construction, the `_drop_pytorch_model` handling, the markers, the GPU counters, the transcript
accumulation and the gate. **Nothing about the REDUCTION is re-derived for this model.** If the
numbers transfer, they transfer against an unchanged lever set; if they do not, that is the finding.

Original header of the BERT driver follows, because everything else in it still applies. TWO of its
paragraphs do NOT apply here and are left standing rather than edited, so that the port is visible:
the one about the extended attention mask and the one that counts fifteen sub-graphs. ViT has no
mask, so there is no sub-graph 0 and there are fourteen. Everything those paragraphs say about WHY
the mask had to be traced is still the reason this driver does not invent a shortcut anywhere else.

SHAFT-GPU, per-layer variant: convert, encrypt, move to the device, run and discard ONE sub-model
at a time.

WHY THIS SYSTEM NEEDS IT NOW, and the rule rather than the preference decided it. After
`s4_param_defer` the two pools read

    W_vram   2.08      peak   362,496 kB     down 19.2x from 6,971,392
    W_host  13.17      peak 2,924,220 kB     down 1.7% from 2,952,288, i.e. UNTOUCHED

so rule 3 puts the next step on the HOST pool, and the host object table at that peak names what is
there:

    870.5 MB  29.1%   heap:python3.10+0xd6be1
    799.0 MB  26.7%   heap:libstdc++.so.6.0.29+0xf3269
    433.3 MB  14.5%   file:db5325...            <- mapped, excluded from W by policy
    366.8 MB  12.3%   anon:[heap]               <- located but unnamed, not targetable

The two rankable rows are 55.8% of the peak between them, and they are the ONNX conversion of the
whole model: a Python-side graph and a libstdc++-side protobuf, alive at the same instant. Four
VRAM levers have run past that object without touching it, because none of them is above CrypTen.

DONOR: SHAFT-CPU's `s6_per_layer` (measured off-path first as `x1_per_layer`), which is the same
mechanism against the same object on the same model. There it took the host peak from 2,056,996 kB
to about 1,290,000, and it did so after that line had spent a day proving the object could not be
reached any other way: three separate releases during the conversion phase -- the plaintext model,
the same with malloc_trim(0), and the protobuf initializers -- each freed a measured 433 MB and
moved the peak by NOTHING. A program that never converts the whole model does not have that peak.

WHAT IS PORTED AND WHAT IS NOT. The sub-model wrappers, the loop structure and the transcript
summation are SHAFT-CPU's `bert_perlayer_cpu.py` verbatim; the seeding, the marker format, the
`GPU|` line, the GPUPEAK print and the gate are `bert_instrumented_gpu.py`'s verbatim. The only
genuinely new line is `.cuda()` per sub-model instead of once for the whole network.

WHAT IS DELIBERATELY NOT ADDED: `torch.cuda.empty_cache()` between layers. It would almost
certainly lower `peak_reserved_kb`, and it would do so by changing the allocator's behaviour rather
than the program's demand -- the same objection that retired jemalloc from this campaign. If the
caching allocator's reserve is worth attacking it must be its own declared step with its own cost.

WHY IT NEEDS PHASE B, stated before the number arrives, and it is the donor's argument unchanged.
Converting and encrypting layer i+1 while layer i's result is already computed interleaves layer
i+1's PRZS draws with layer i's Beaver draws. The mask stream therefore differs, fixed-point
truncation error is share-dependent, and the exact gate cannot separate that from a broken
protocol. The two phase B conditions carry it instead:

  (a) THE TRANSCRIPT. The same bytes must cross the wire. The counters are summed over the fifteen
      sub-graphs and printed as integers, because CrypTen's own two-decimal print has 5 MB of
      resolution and "the same bytes" cannot rest on that. On SHAFT-CPU this condition came back
      not merely within tolerance but IDENTICAL, bytes and round count both, which is the outcome
      to expect here.
  (b) THE PLAINTEXT EQUIVALENCE, `impl/equiv_per_layer.py` in the donor's tree, which compares
      whole-model against per-layer inference outside the protocol where it must be exact. It is
      the same decomposition of the same model, so it carries over; it is re-run in this image
      rather than cited, because "it passed elsewhere" is not a measurement of this build.

THE ATTENTION MASK IS ITS OWN SUB-GRAPH, and the reason is a measurement rather than a preference.
The whole-model graph takes `attention_mask` as an input and computes the extended mask inside.
Split apart, that computation has no home, and the first version of this driver did what the
SHAFT-CPU port does: built it directly as `ct.cryptensor(torch.zeros(1, 1, 1, max_length))`. The
VALUE is right, and `equiv_per_layer.py` checks it against the model's own method rather than
believing it. The TRANSCRIPT was not:

    s4_param_defer (whole model)   bytes=11,229,212,672   rounds=1526
    first attempt (decomposed)     bytes=11,228,688,384   rounds=1525

524,288 bytes and one round FEWER -- 0.0047% of the transcript, and phase B requires the bytes to be
IDENTICAL. A shortcut that skips protocol work is what that condition exists to catch even when it
skips in the cheaper direction, so the shortcut is gone: `ExtendedMaskWrapper` traces the model's own
`get_extended_attention_mask` as sub-graph 0, and there are fifteen sub-graphs rather than fourteen.

On SHAFT-CPU the same shortcut cost nothing (1495 rounds against 1495 and identical bytes), which is
why it went unnoticed there and why that line's port keeps it. On this device the mask branch is not
free, and the two lines differ in their transcripts for that reason: 11,228,688,384 on CPU against
11,229,212,672 on GPU, at the same layer count and sequence length.

Marker format on stderr, unchanged from `bert_instrumented_gpu.py`:
    MEM|<party>|<layer>|<phase>|<op>|<rss_kb>|<timestamp_ms>|<vmhwm_kb>
    GPU|<party>|<layer>|<phase>|<op>|<cuda_alloc_kb>|<cuda_reserved_kb>|<timestamp_ms>
"""

import gc
import os
import sys
import time

import torch
import torch.nn as nn

import crypten  # noqa: F401  (imported before the levers, exactly as the baseline does)
import graphop_markers_gpu  # noqa: F401  measurement-neutral per-node markers
import embed_chunk_gpu  # noqa: F401  chunks the word-embedding matmul (no-op unless CHUNKS>1)

# THE ENV-GATED LEVER IMPORTS ARE THE BASELINE'S, VERBATIM, AND DROPPING THEM COST A RUN (18.08.).
# The first version of this port loaded only the `SHAFT_LEVERS` modules, on the assumption that the
# declaration listed everything a step carries. It does not: `bert_instrumented_gpu.py` imports
# these at module level from their own environment variables, and `s4_param_defer`'s declaration
# accordingly lists only `param_spill_gpu,param_defer_gpu` while relying on `SHAFT_LIMB_LOOP=1` and
# `SHAFT_EMBED_CHUNKS=16` to bring in the rest. Without this block the port ran to completion,
# exit 0, gate and transcript fine, and reported VRAM 6,014,976 kB against `s4`'s 362,496 -- a 16.6x
# regression that was really three missing levers.
#
# The baseline's own comment records the same failure from the other direction: "SHAFT_LIMB_LOOP=1
# ran to completion with a peak BIT-IDENTICAL to the baseline's, because nothing imported the
# module." Each module announces itself on stderr when it patches and the harness fails the run when
# a requested announcement is missing, which is the check that catches this class.
#
# They are imported ONLY when their variable asks for them, for the baseline's reason: they patch
# CrypTen unconditionally at import time, so importing a switched-off lever still wraps every
# MatMul.forward in an extra Python frame.
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
    """Current resident size and the kernel's exact high-water mark, from ONE read.

    VmHWM is what BRACKETS the peak. This system's host peak is a transient: the campaign poller
    reads it 18 to 34% low, so a composition sampled "at the peak" describes a lower moment, and no
    poll interval fixes that. VmHWM is monotone and exact, so emitting it beside every phase marker
    localises the peak to the interval between two named markers with no sampling risk at all.
    """
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


# --------------- Sub-model wrappers for ONNX tracing ---------------
# SHAFT-CPU's `bert_perlayer_cpu.py` verbatim, which took them from the retired campaign's
# `02_per_layer`. Each exists because `from_pytorch` traces a module with a fixed signature, and a
# bare `BertLayer` returns a tuple.

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
    """ViT's head: the final LayerNorm, then the classifier on the CLS token.

    `ViTForImageClassification.forward` reads `sequence_output[:, 0, :]` after `vit.layernorm`, so
    the slice belongs here rather than being applied to the result. Written out rather than reused
    from the model because the model container is dropped before this point, and the arithmetic is
    two lines that the equivalence test checks against the model's own output.
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


def _comm_report(tag, b, r):
    """Per-sub-graph transcript, so a missing round can be LOCALISED rather than reasoned about.

    The summed transcript of this port came back 524,288 bytes and one round below `s4_param_defer`,
    and the same difference exists between the two systems' BASELINES (CPU 11,228,688,384 / 1495
    against GPU 11,229,212,672 / 1496) with no lever on either. So one protocol call fires in the
    GPU whole-model graph and in neither the CPU whole-model graph nor this decomposition. Guessing
    which one cost two wrong hypotheses; this prints the answer.
    """
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
    # The list entries are replaced with placeholders so `del model` frees everything that is not
    # one of the pieces below, rather than leaving the whole graph alive through one reference.
    # NO `bert_stub` AND NO `model_dtype`: both existed only to trace the extended attention mask
    # through the model's own code path, and ViT has no attention mask at all. See the header.
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

    # COUNTED, NOT COMPUTED, and the first version of this file got it wrong. It inherited the BERT
    # driver's `subgraphs={n_layers + 3}`, where the 3 is mask + embedding + head. ViT has no mask,
    # so the correct constant is `n_layers + 2`, and the run printed `subgraphs=15` for a
    # decomposition that has fourteen. Nothing measured was affected -- the bytes and rounds are
    # summed from the graphs themselves -- but the label was a claim about the structure and it was
    # false. Counting removes the constant instead of correcting it, which is the same fix the
    # sbatch's `n12_seq128` needed on the same day and for the same reason.
    comm_bytes, comm_rounds, n_subgraphs = 0, 0, 0
    t0 = time.time()

    # ==== Phase 0 DOES NOT EXIST HERE ====
    # The BERT driver traces the extended attention mask as its own sub-graph, and its header
    # explains at length why it must be traced rather than built as zeros: skipping it cost
    # 524,288 bytes and one round, and phase B requires the bytes to be identical. **ViT has no
    # attention mask**, so there is nothing to trace and nothing to skip. Fourteen sub-graphs
    # follow, against the BERT driver's fifteen, and the counter below sums over all of them
    # exactly as it does there.

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

    # THE TRANSCRIPT, summed over the sub-graphs and printed as integers. The baseline reads the
    # same attribute from its single graph, so the two numbers are the same quantity measured over
    # the same protocol; what phase B asks is whether they are EQUAL.
    #
    # `subgraphs` is the number actually accumulated above. A hard stop rather than a print, because
    # a decomposition that traced a different number of pieces than it thinks it did is exactly the
    # kind of plausible artefact this campaign keeps paying for.
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
        # The gate observable, identical in form to the baseline's: the decrypted output tensor at
        # full precision, not the predicted label, which is one bit and would pass for two runs
        # that computed materially different logits.
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
