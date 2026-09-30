#!/usr/bin/env python3
"""
SHAFT-CPU baseline: BERT-base under 2-party CrypTen MPC, both parties on the host CPU.

Derived from operator-bench/shaft/00_default/bert_instrumented.py. Three things differ from that
file, and all three are SETUP rather than levers: they are identical for the baseline and for
every step, exactly as the pinned thread count is.

  1. THE SEEDS ARE PINNED, immediately after ct.init(). Without this the system has no
     correctness gate at all (see below).
  2. THE DECRYPTED LOGITS ARE PRINTED as `GATE|logits|...`. The original printed only
     `Prediction: 0`, which is one bit.
  3. THE IN-PROCESS VmRSS POLLER IS REMOVED. The campaign poller measures this process from
     outside, and the in-process one wrote `party<N>_vmrss.log` into the results directory --
     a filename `make_steps.py` globs as a poll trace. On SHAFT-GPU that turned one run into two
     apparent "replicates" measured through different channels (VmHWM against a polled maximum),
     and their 39% disagreement was published as a spread. Removing it means one run produces one
     trace, from one instrument.

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

    model_name = os.environ.get("SHAFT_MODEL", "andeskyl/bert-base-cased-sst2")
    max_length = int(os.environ.get("SHAFT_MAX_LENGTH", "128"))
    n_layers = int(os.environ.get("SHAFT_N_LAYERS", "12"))
    print(
        f"CONFIG|party={rank}|n_layers={n_layers}|seq_len={max_length}|model={model_name}",
        file=sys.stderr, flush=True,
    )

    mem("main", "after_init")

    # ---- Load plaintext model ----
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Optionally truncate to fewer layers for quick tests
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        actual_layers = len(model.bert.encoder.layer)
        if n_layers < actual_layers:
            model.bert.encoder.layer = model.bert.encoder.layer[:n_layers]
            print(f"Truncated model to {n_layers} layers (was {actual_layers})",
                  file=sys.stderr, flush=True)

    mem("main", "after_model_load")

    # ---- Convert to CrypTen graph ----
    dummy_input_ids = torch.zeros(1, max_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(1, max_length, dtype=torch.long)

    private_model = ct.nn.from_pytorch(
        model, (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
    )
    mem("main", "after_model_convert")

    # ---- Encrypt model ----
    private_model.encrypt()
    mem("main", "after_model_encrypt")

    # ---- Create and encrypt input ----
    input_ids = torch.randint(0, 28996, (1, max_length))
    attention_mask = torch.ones(1, max_length, dtype=torch.long)
    token_type_ids = torch.zeros(1, max_length, dtype=torch.long)

    inputs_enc = ct.cryptensor(input_ids).to(device)
    attention_mask_enc = ct.cryptensor(attention_mask).to(device)
    token_type_enc = ct.cryptensor(token_type_ids).to(device)
    mem("main", "after_input_encrypt")

    # ---- Run inference ----
    t0 = time.time()
    with ct.no_grad():
        outputs_enc = private_model(inputs_enc, attention_mask_enc, token_type_enc)
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
