#!/usr/bin/env python3
"""
SHAFT-GPU ViT transfer, BASELINE: ViT-Base/16 under 2-party CrypTen MPC on an H200.

`bert_instrumented_gpu.py` with the model replaced and nothing else. It is the GPU twin of
SHAFT-CPU's `vit_instrumented_cpu.py`, and it exists because SHAFT-CPU's transfer answers the
question on the HOST pool only: this line's reduction is 19.232x on VRAM against 1.365x on host, so
"does the BERT-derived reduction carry to another architecture" has a second and unmeasured answer
on the device pool. No other two-pool system in the campaign loads a model, so no other line can ask
it (`GEOMETRY-TRANSFER.md`).

WHAT THE ARCHITECTURE FORCES, and it is only what the model forces:

  1. `ViTForImageClassification` instead of `AutoModelForSequenceClassification`.
  2. ONE input, `pixel_values` of shape (1, 3, 224, 224), instead of three token tensors. So
     `from_pytorch` is traced with a one-element tuple and the encrypted call takes one argument.
  3. The layer list is `model.vit.encoder.layer`, not `model.bert.encoder.layer`.
  4. The checkpoint comes from a DIRECTORY rather than a hub id, because the measured image bakes in
     only the BERT weights and the compute nodes have no network. `SHAFT_MODEL` points at a
     bind-mounted `/vit`. That is SETUP and is identical for the baseline and the endpoint, exactly
     as the pinned seed is.

WHAT IS IDENTICAL, deliberately and line for line: the lever loader, the seeds, the markers, the
transcript counters, the GPU counters and the gate. The gate observable is the decrypted output
tensor at full precision, here 1000 ImageNet logits rather than 2 sentiment logits: a longer string
and the same claim.

WHAT THIS CANNOT SHOW: it is a transfer CHECK, not a second waterfall. Baseline and reduced endpoint
only, so the comparison is "does the same lever set still work", never "is this the best line for
ViT".

Original header of the BERT driver follows, because everything in it still applies.

GPU-instrumented BERT-base benchmark for SHAFT (CrypTen-based) on PALMA H200 / H200mini.

Same 2-party CrypTen setup as the CPU default run, but the encrypted model and
inputs are moved to CUDA. In addition to host VmRSS polling, this version records
the per-party peak GPU memory (both torch's allocated and reserved counters), which
is the number this smoke test exists to produce: nobody publishes VRAM for SNNI.

Marker formats (stderr):
    MEM|<party>|<layer>|<phase>|<op>|<rss_kb>|<ts_ms>|<vmhwm_kb>
    GPU|<party>|<phase>|<op>|<cuda_alloc_kb>|<cuda_reserved_kb>|<ts_ms>

The trailing VmHWM field was added 2026-08-01, in the same re-measurement as the move off the
MIG slice, so no published row mixes the two marker formats. It is appended rather than inserted
because the sbatch reads the RSS field positionally. See read_rss_hwm_kb for why it exists.
"""

import os
import sys
import time
import threading
import torch
import crypten
import graphop_markers_gpu  # measurement-neutral per-node markers (host + GPU), CPU-consistent
import embed_chunk_gpu  # chunk the word-embedding matmul (no-op unless SHAFT_EMBED_CHUNKS>1)

# Lever modules are imported ONLY when their env var asks for them. They patch CrypTen
# unconditionally at import time and rely on an internal no-op check, so importing one that is
# switched off would still wrap every MatMul.forward in an extra Python frame -- the baseline
# would then be running a lever's code path while reporting itself as the baseline.
#
# Each module announces itself on stderr when it patches, and the harness fails the run if the
# announcement is missing. That check found this: SHAFT_LIMB_LOOP=1 ran to completion with a peak
# BIT-IDENTICAL to the baseline's, because nothing imported the module. Without the check it would
# have been published as "this lever reduces nothing", which is a claim about the lever rather
# than about the wiring.
if os.environ.get("SHAFT_LIMB_LOOP", "0") == "1":
    import matmul_limb_loop_gpu  # noqa: F401
if int(os.environ.get("SHAFT_MATMUL_CHUNKS", "1")) > 1:
    import matmul_chunk_gpu  # noqa: F401
if os.environ.get("SHAFT_PLAINTEXT_FREE", "0") == "1":
    import plaintext_free_gpu  # noqa: F401

# GENERIC LEVER SLOT. `SHAFT_LEVERS=mod_a,mod_b` imports those modules here and nothing else does.
#
# It exists so that adding a lever stops being a change to THIS file. Every lever above needed its
# own import line, and this file's md5 is the identity of the measured program: changing it means
# the baseline a later step is gated against was produced by different code, so the whole line has
# to be re-measured for what is, in behaviour, no change at all. One re-measurement bought that
# once; from here a new lever is a new module and an environment variable.
#
# Each module decides from its own environment variable whether to patch, and announces itself on
# stderr when it does. The harness fails the run when a requested announcement is missing, because
# a lever that never loaded is a successful run with a zero delta and no error.
for _lever in os.environ.get("SHAFT_LEVERS", "").split(","):
    _lever = _lever.strip()
    if _lever:
        __import__(_lever)

import crypten as ct

DEVICE = os.environ.get("SHAFT_DEVICE", "cuda")


def read_vmrss_kb():
    return read_rss_hwm_kb()[0]


def read_rss_hwm_kb():
    """Current resident size and the kernel's exact high-water mark, from ONE read.

    VmHWM is what BRACKETS the peak. This system's host peak is a transient: the campaign
    poller reads it 18 to 34% low, so a composition sampled "at the peak" describes a lower
    moment, and no poll interval fixes that (peak WIDTH varies by step). VmHWM is monotone
    and exact, so emitting it beside every phase marker localises the peak to the interval
    between two named markers with no sampling risk at all -- which is a question the poller
    cannot answer and this can, for the cost of parsing one more line of a file already open.
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
    rss, hwm = read_rss_hwm_kb()
    print(f"MEM|{rank}|{layer_idx}|{phase}|{op}|{rss}|{ts}|{hwm}",
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


def main():
    ct.init()
    rank = ct.comm.get().get_rank()

    # DETERMINISTIC SEEDING, and it lives HERE rather than in multiprocess_launcher.py because
    # only this script is bind-mounted into the container; the launcher runs from the image's own
    # baked-in copy, so patching it on the host has no effect at all.
    #
    # Why it is needed: ct.init() draws its three generator seeds from os.urandom(8)
    # (crypten/__init__.py:214-221), so the secret shares differ every run. Fixed-point truncation
    # error is probabilistic, so the reconstructed logits then differ by far more than the "last
    # bits" the MANIFEST's 1e-6 tolerance was written for. Five unfrozen runs of this identical
    # image produced 0.2032, 0.2348, 0.5051, 0.6427 and 0.7734 -- a 3.8x spread. A gate compares a
    # step against the BASELINE's output, so with that spread this system had no usable gate, and
    # the defect stayed invisible because s0 IS the baseline and s1 was never published.
    #
    # The seed is part of the measured system, applied identically to the baseline and every step,
    # as thread count is pinned rather than inherited. It exists so the gate can exist; it is not
    # how the protocol would be deployed, and the MANIFEST says so.
    #
    # manual_seed() refuses outside debug mode, so debug is enabled only for that call and
    # restored immediately, before any measured work happens.
    from crypten.config import cfg as _cfg
    _dbg = _cfg.debug.debug_mode
    _cfg.debug.debug_mode = True
    ct.manual_seed(0xDEADBEEF + rank, 0xC0FFEE + rank, 0x5EED)
    _cfg.debug.debug_mode = _dbg
    torch.manual_seed(0x5EED)
    print(f"SEEDED|party={rank}|next=0x{0xDEADBEEF + rank:x} global=0x5eed",
          file=sys.stderr, flush=True)

    use_cuda = DEVICE == "cuda" and torch.cuda.is_available()
    if DEVICE == "cuda" and not use_cuda:
        print(f"FATAL|party={rank}|torch.cuda.is_available()==False, "
              f"cannot run GPU test", file=sys.stderr, flush=True)
        sys.exit(2)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPUINFO|party={rank}|device={DEVICE}|name={gpu_name}|"
              f"visible={os.environ.get('CUDA_VISIBLE_DEVICES','?')}",
              file=sys.stderr, flush=True)

    pid = os.getpid()
    results_dir = os.environ.get("RESULTS_DIR", "/results/shaft")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"party{rank}.pid"), "w") as f:
        f.write(str(pid))

    # THE IN-PROCESS VmRSS POLLER IS REMOVED, and that is a measurement fix rather than a lever.
    # It wrote `party<N>_vmrss.log` into the results directory, and `make_steps.py` globs exactly
    # that filename as a poll trace. Each run therefore appeared TWICE -- once through the
    # campaign's external poller and once through this one -- so three replicates were counted as
    # six, and the two channels disagreed by 46.6% because one reports a polled maximum and the
    # other a VmHWM. One run must produce one trace, from one instrument. SHAFT-CPU removed the
    # same call for the same reason.
    #   start_vmrss_poller(rank, interval_ms=100, output_dir=results_dir)

    model_name = os.environ.get("SHAFT_MODEL", "/vit")
    # `max_length` is NOT a free parameter here the way a sequence length is: the token count
    # follows from image_size and patch_size, so it is derived from the config below and reported
    # rather than accepted from an environment value that could contradict the checkpoint.
    max_length = int(os.environ.get("SHAFT_MAX_LENGTH", "197"))
    n_layers = int(os.environ.get("SHAFT_N_LAYERS", "12"))
    print(f"CONFIG|party={rank}|n_layers={n_layers}|seq_len={max_length}|"
          f"model={model_name}|device={DEVICE}", file=sys.stderr, flush=True)

    mem("main", "after_init")

    from transformers import ViTForImageClassification

    # NO TOKENIZER. A vision model has an image processor instead, and it is not used: the measured
    # workload feeds a fixed tensor rather than a decoded image, exactly as the BERT driver feeds
    # `torch.randint` token ids rather than a tokenised sentence. What is measured is the protocol,
    # not the preprocessing.
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()

    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        actual_layers = len(model.vit.encoder.layer)
        if n_layers < actual_layers:
            model.vit.encoder.layer = model.vit.encoder.layer[:n_layers]

    cfg = model.config
    _img = cfg.image_size
    _tok = (cfg.image_size // cfg.patch_size) ** 2 + 1
    print(f"VIT|image={cfg.image_size}|patch={cfg.patch_size}|tokens={_tok}|hidden={cfg.hidden_size}"
          f"|heads={cfg.num_attention_heads}|ffn={cfg.intermediate_size}",
          file=sys.stderr, flush=True)
    if _tok != max_length:
        # A hard stop rather than a warning. The token count is what this whole probe is about, and
        # a run whose configuration disagrees with its checkpoint would publish a plausible number
        # for a geometry nobody chose. Same discipline as MOAI-GPU's SNNI_GEOM abort.
        sys.exit(f"FAILED: checkpoint has {_tok} tokens, SHAFT_MAX_LENGTH says {max_length}")

    mem("main", "after_model_load")

    dummy_pixel_values = torch.zeros(1, cfg.num_channels, _img, _img)

    private_model = ct.nn.from_pytorch(model, (dummy_pixel_values,))
    mem("main", "after_model_convert")

    private_model.encrypt()
    mem("main", "after_model_encrypt")

    # Move the encrypted model to GPU.
    if use_cuda:
        private_model.cuda()
        mem("main", "after_model_cuda")

    # A FIXED pseudo-image rather than a decoded photograph, drawn from the pinned generator so the
    # gate reproduces. The protocol's cost does not depend on the values, but a degenerate input can
    # make the gate uninformative, which is the failure SIGMA-GPU's all-zero observable had.
    pixel_values = torch.randn(1, cfg.num_channels, _img, _img)

    inputs_enc = ct.cryptensor(pixel_values).to(DEVICE)
    mem("main", "after_input_encrypt")

    t0 = time.time()
    with ct.no_grad():
        outputs_enc = private_model(inputs_enc)
    t1 = time.time()
    mem("main", "after_inference")

    outputs = outputs_enc.get_plain_text()
    mem("main", "after_decrypt")

    # THE TRANSCRIPT, exact rather than rounded. CrypTen's own Graph prints
    # "comm byte: 10.46 GB, round: 1496" at two decimals, which is 5 MB of resolution and cannot
    # support the claim "the same bytes". PROCEDURE.md's phase B rests on that claim, so the
    # counters are published as integers, from the same run as the peak and the gate.
    _tb = getattr(private_model, "total_comm_bytes", None)
    _tr = getattr(private_model, "total_comm_rounds", None)
    if _tb is None:
        for _m in getattr(private_model, "_modules", {}).values():
            if hasattr(_m, "total_comm_bytes"):
                _tb, _tr = _m.total_comm_bytes, _m.total_comm_rounds
                break
    print(f"TRANSCRIPT|party={rank}|bytes={_tb if _tb is not None else 'NA'}|"
          f"rounds={_tr if _tr is not None else 'NA'}", flush=True)

    if use_cuda:
        peak_alloc = int(torch.cuda.max_memory_allocated() / 1024)
        peak_res = int(torch.cuda.max_memory_reserved() / 1024)
        print(f"GPUPEAK|party={rank}|peak_alloc_kb={peak_alloc}|"
              f"peak_reserved_kb={peak_res}", file=sys.stderr, flush=True)

    if rank == 0:
        predictions = outputs.argmax(dim=-1)
        print(f"Prediction: {predictions.item()}", flush=True)
        # Correctness gate observable. The predicted LABEL is one bit and would pass for two runs
        # that computed materially different logits, so the gate is the decrypted output tensor
        # itself, printed at full precision. Every lever in this campaign changes only when
        # memory is held, never what is computed, so these values must not move; a step that
        # moves them has changed the protocol rather than its memory and its number is void.
        #
        # This lives in the BASELINE script, not in any lever, and therefore applies identically
        # to every step. A gate that differed between steps would not be a gate.
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
