#!/usr/bin/env python3
"""SHAFT-GPU ViT transfer baseline: ViT-Base/16 under 2-party CrypTen MPC on an H200.
= bert_instrumented_gpu.py with the model replaced (ViTForImageClassification, one pixel_values
input, model.vit.encoder.layer, checkpoint from a bind-mounted /vit). Transfer check on the device
pool (this line: 19.232x VRAM, 1.365x host). Lever loader, seeds, markers, transcript, GPU counters
and gate as in the BERT driver."""

import os
import sys
import time
import threading
import torch
import crypten
import graphop_markers_gpu  # measurement-neutral per-node markers (host + GPU), CPU-consistent
import embed_chunk_gpu  # chunk the word-embedding matmul (no-op unless SHAFT_EMBED_CHUNKS>1)

# Lever modules are imported only when their env var asks; each patches CrypTen at import time and
# announces on stderr, and the harness fails the run if a requested announcement is missing.
if os.environ.get("SHAFT_LIMB_LOOP", "0") == "1":
    import matmul_limb_loop_gpu  # noqa: F401
if int(os.environ.get("SHAFT_MATMUL_CHUNKS", "1")) > 1:
    import matmul_chunk_gpu  # noqa: F401
if os.environ.get("SHAFT_PLAINTEXT_FREE", "0") == "1":
    import plaintext_free_gpu  # noqa: F401

# GENERIC LEVER SLOT. `SHAFT_LEVERS=mod_a,mod_b` imports those modules here, so adding a lever is a
# new module + env var rather than an edit to this file (whose md5 is the measured program's identity).
for _lever in os.environ.get("SHAFT_LEVERS", "").split(","):
    _lever = _lever.strip()
    if _lever:
        __import__(_lever)

import crypten as ct

DEVICE = os.environ.get("SHAFT_DEVICE", "cuda")


def read_vmrss_kb():
    return read_rss_hwm_kb()[0]


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

    # DETERMINISTIC SEEDING, here (not in the launcher) because only this script is bind-mounted.
    # ct.init() seeds from os.urandom, so without a fixed seed the shares and reconstructed logits
    # differ every run and the gate is unusable; the seed is part of the measured system (MANIFEST).
    # manual_seed() refuses outside debug mode, so debug is toggled on only for that call.
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

    # The in-process VmRSS poller is removed: make_steps.py globs its `party<N>_vmrss.log` as a
    # poll trace, so each run was counted twice and the two channels disagreed by 46.6%.
    #   start_vmrss_poller(rank, interval_ms=100, output_dir=results_dir)

    model_name = os.environ.get("SHAFT_MODEL", "/vit")
    # max_length is not free here: the token count follows from image_size and patch_size, so it is
    # derived from the config below and checked against this value rather than accepted from it.
    max_length = int(os.environ.get("SHAFT_MAX_LENGTH", "197"))
    n_layers = int(os.environ.get("SHAFT_N_LAYERS", "12"))
    print(f"CONFIG|party={rank}|n_layers={n_layers}|seq_len={max_length}|"
          f"model={model_name}|device={DEVICE}", file=sys.stderr, flush=True)

    mem("main", "after_init")

    from transformers import ViTForImageClassification

    # No tokenizer: the workload feeds a fixed tensor, not a decoded image (as the BERT driver feeds
    # random token ids); what is measured is the protocol, not the preprocessing.
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
        # Hard stop: a run whose configuration disagrees with its checkpoint would publish a
        # plausible number for a geometry nobody chose (as MOAI-GPU's SNNI_GEOM abort guards).
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

    # A fixed pseudo-image from the pinned generator so the gate reproduces; a degenerate input can
    # make the gate uninformative (as SIGMA-GPU's all-zero observable did).
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

    # Transcript counters published as integers (CrypTen's Graph prints 2 decimals = 5 MB
    # resolution), from the same run as the peak and gate, to support PROCEDURE.md phase B.
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
        # Correctness gate: the decrypted output tensor at full precision (the predicted label is
        # one bit, too coarse). In the baseline so it applies to every step; values must not move.
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
