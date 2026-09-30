#!/usr/bin/env python3
"""
SHAFT-CPU ViT transfer, baseline: ViT-Base/16 under 2-party CrypTen MPC, both parties on the host.

**THIS IS THE ONLY REAL ARCHITECTURE CHANGE IN THE CAMPAIGN'S TRANSFER CHECK**, and the reason is
in `GEOMETRY-TRANSFER.md`: BumbleBee, SHARK, SIGMA-GPU and MOAI-GPU compute against shape
parameters and load no model at all, so for them "ViT" would be an integer. This system loads and
traces a real model, so here the question the README asks -- does the BERT-derived reduction carry
to another architecture -- can actually be answered.

WHAT DIFFERS FROM `bert_instrumented_cpu.py`, and it is only what the model forces:

  1. `ViTForImageClassification` instead of `AutoModelForSequenceClassification`.
  2. ONE input, `pixel_values` of shape (1, 3, 224, 224), instead of three token tensors. So
     `from_pytorch` is traced with a one-element tuple and the encrypted call takes one argument.
  3. The layer list is `model.vit.encoder.layer`, not `model.bert.encoder.layer`.
  4. The model comes from a DIRECTORY rather than a hub id, because the measured image bakes in
     only the BERT checkpoint and the compute nodes have no network. `SHAFT_MODEL` therefore points
     at a bind-mounted `/vit` and `transformers` loads it as a local path. That is a change to the
     SETUP and it is identical for the baseline and for the endpoint, exactly as the pinned seed is.

WHAT IS IDENTICAL, deliberately and line for line: the lever loader, the seeds, the markers, the
transcript counters and the gate. The gate observable is the decrypted output tensor at full
precision, which for this model is 1000 ImageNet logits rather than 2 sentiment logits; that is a
longer string and the same claim.

WHAT THIS CANNOT SHOW: it is a transfer CHECK, not a second waterfall. Baseline and the reduced
endpoint only, so the comparison is "does the same lever set still work", never "is this the best
line for ViT".

Marker format on stderr, unchanged:
    MEM|<party>|<layer>|<phase>|<op>|<rss_kb>|<timestamp_ms>
"""

import importlib
import os
import sys
import time
import torch
import crypten
import crypten as ct

# LEVERS ARE LOADED BY NAME FROM THE ENVIRONMENT, and only when asked for.
#
# `SHAFT_LEVERS=a,b` imports modules a and b, which are bind-mounted beside this file. Nothing is
# imported for the baseline, so the baseline runs no lever's code path -- a module that patches at
# import time would wrap the patched function in an extra Python frame even when its own no-op
# check is true, and the baseline would then be measuring a lever while reporting itself as the
# baseline.
#
# It is a NAME LIST rather than one `if` per lever so that adding a lever never edits this file.
# This script is the gate observable's home and is bind-mounted identically into every run of
# every step; a step that had to edit it would change the baseline as well.
#
# Each module announces itself as `LEVER|<name>|patched` when it patches, and the sbatch fails the
# run if the announcement is missing for a requested lever. Without that check a lever that never
# imported is a successful run with a zero delta and no error, i.e. the baseline published under
# the lever's name. The marker appears EXACTLY when the module patched, never in the no-op case:
# a marker that prints either way reads as confirmation while proving nothing.
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


def main():
    ct.init()
    device = "cpu"
    rank = ct.comm.get().get_rank()

    # DETERMINISTIC SEEDING, and it lives HERE rather than in multiprocess_launcher.py because
    # only this script and the lever modules are bind-mounted into the container; the launcher
    # runs from the image's own baked-in copy, so patching it on the host has no effect at all.
    # That cost SHAFT-GPU a debugging round already.
    #
    # Why it is needed: ct.init() draws its three generator seeds from os.urandom(8)
    # (crypten/__init__.py:214-221), so the secret shares differ on every run. Fixed-point
    # truncation error is probabilistic, so the reconstructed logits then move far more than the
    # "last bits" a 1e-6 tolerance is written for. Measured on the GPU sibling, which is the same
    # code path: five runs of one image spanned 3.8x on the first logit and one produced
    # -3.47e+09, a truncation wraparound.
    #
    # A gate compares a step against the BASELINE's output, so with that spread the gate can never
    # pass and the system has no gate at all. The defect is silent because the baseline DEFINES
    # the reference rather than being compared against it.
    #
    # The seed is part of the measured system, applied identically to the baseline and to every
    # step, exactly as OMP_NUM_THREADS is pinned rather than inherited. It exists so the gate can
    # exist. It is NOT how the protocol would be deployed -- fixed shares would be a security
    # defect in a real deployment -- and the MANIFEST says so.
    #
    # manual_seed() refuses outside debug mode, so debug is enabled for that one call and restored
    # immediately, before any measured work happens.
    from crypten.config import cfg as _cfg
    _dbg = _cfg.debug.debug_mode
    _cfg.debug.debug_mode = True
    ct.manual_seed(0xDEADBEEF + rank, 0xC0FFEE + rank, 0x5EED)
    _cfg.debug.debug_mode = _dbg
    torch.manual_seed(0x5EED)
    # The marker exists so that a seed which silently failed to take cannot be mistaken for one
    # that worked. The sbatch fails the run if it is absent from both parties.
    print(f"SEEDED|party={rank}|next=0x{0xDEADBEEF + rank:x}|"
          f"local=0x{0xC0FFEE + rank:x}|global=0x5eed|torch=0x5eed",
          file=sys.stderr, flush=True)

    # Write PID for external monitoring. The campaign poller reads these files rather than
    # matching a cmdline: CrypTen starts its parties by SPAWN, and a spawned child's cmdline is
    # `python3 -c from multiprocessing.spawn import spawn_main`, which never contains the script
    # name, so a cmdline filter cannot find the workers even in principle.
    pid = os.getpid()
    results_dir = os.environ.get("RESULTS_DIR", "/results/shaft")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"party{rank}.pid"), "w") as f:
        f.write(str(pid))

    model_name = os.environ.get("SHAFT_MODEL", "/vit")
    max_length = int(os.environ.get("SHAFT_MAX_LENGTH", "197"))
    n_layers = int(os.environ.get("SHAFT_N_LAYERS", "12"))
    print(
        f"CONFIG|party={rank}|n_layers={n_layers}|seq_len={max_length}|model={model_name}",
        file=sys.stderr, flush=True,
    )

    mem("main", "after_init")

    # ---- Load plaintext model ----
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
            print(f"Truncated model to {n_layers} layers (was {actual_layers})",
                  file=sys.stderr, flush=True)
    cfg = model.config
    _tok = (cfg.image_size // cfg.patch_size) ** 2 + 1
    print(f"VIT|image={cfg.image_size}|patch={cfg.patch_size}|tokens={_tok}|hidden={cfg.hidden_size}"
          f"|heads={cfg.num_attention_heads}|ffn={cfg.intermediate_size}",
          file=sys.stderr, flush=True)

    mem("main", "after_model_load")

    # ---- Convert to CrypTen graph ----
    # ONE input. `max_length` is not a free parameter here the way a sequence length is: the token
    # count follows from image_size and patch_size, so the driver derives the input shape from the
    # config and reports the token count rather than accepting a contradictory environment value.
    _img = model.config.image_size
    dummy_pixel_values = torch.zeros(1, model.config.num_channels, _img, _img)

    private_model = ct.nn.from_pytorch(model, (dummy_pixel_values,))
    mem("main", "after_model_convert")

    # ---- Encrypt model ----
    private_model.encrypt()
    mem("main", "after_model_encrypt")

    # ---- Create and encrypt input ----
    # A FIXED pseudo-image rather than a decoded photograph, drawn from the pinned generator so the
    # gate reproduces. Normalised to roughly the processor's range so the activations are not
    # pathological; the protocol's cost does not depend on the values, but a degenerate input can
    # make the gate uninformative, which is the failure SIGMA-GPU's all-zero observable had.
    pixel_values = torch.randn(1, model.config.num_channels, _img, _img)

    inputs_enc = ct.cryptensor(pixel_values).to(device)
    mem("main", "after_input_encrypt")

    # ---- Run inference ----
    t0 = time.time()
    with ct.no_grad():
        outputs_enc = private_model(inputs_enc)
    t1 = time.time()
    mem("main", "after_inference")

    # ---- Decrypt output ----
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

    if rank == 0:
        predictions = outputs.argmax(dim=-1)
        print(f"Prediction: {predictions.item()}", flush=True)
        # THE CORRECTNESS GATE OBSERVABLE. The predicted LABEL is one bit and would pass for two
        # runs whose logits differed materially, which is exactly the failure this gate exists to
        # catch, so the observable is the decrypted output tensor itself at full precision.
        #
        # It lives in the BASELINE script, so it applies identically to every step. A gate that
        # differed between steps would not be a gate.
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
