# Census variant of bert_instrumented_cpu.py (only the census call in mem() differs); regenerate, do not edit.
#!/usr/bin/env python3
"""SHAFT-CPU baseline: BERT-base, 2-party CrypTen MPC, both parties on host CPU.
From shaft/00_default/bert_instrumented.py; setup changes (same at every step): seeds pinned after
ct.init(), logits as GATE|logits|, in-process VmRSS poller removed.
Stderr marker: MEM|<party>|<layer>|<phase>|<op>|<rss_kb>|<timestamp_ms>.
"""

import importlib
import os
import sys
import time
import torch
import crypten
import crypten as ct

# Levers loaded by name from SHAFT_LEVERS (bind-mounted beside this file); none for the baseline so
# it runs no lever code path. Name list so adding a lever never edits this gate-observable script.
# Each module prints LEVER|<name>|patched only when it patches; the sbatch fails if absent.
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
    # Probe only, off by default: census at these eight main-thread phase points (~2s each), joined
    # to the peak via the poll trace. Opt-in via SNNI_PY_CENSUS_PHASES for reading a run's shape.
    if os.environ.get("SNNI_PY_CENSUS_PHASES") == "1":
        try:
            import snni_pycensus
            snni_pycensus.dump(f"{phase}:{op}")
        except Exception:
            pass


def main():
    ct.init()
    device = "cpu"
    rank = ct.comm.get().get_rank()

    # Deterministic seeding, part of the measured system (applied identically to every step). Lives
    # here, not in the launcher, which runs from the image's baked-in copy and is not bind-mounted.
    # Without it ct.init() reseeds from os.urandom per run, logits swing far past the 1e-6 tolerance
    # and the gate can never pass. manual_seed() needs debug mode, toggled for the one call.
    from crypten.config import cfg as _cfg
    _dbg = _cfg.debug.debug_mode
    _cfg.debug.debug_mode = True
    ct.manual_seed(0xDEADBEEF + rank, 0xC0FFEE + rank, 0x5EED)
    _cfg.debug.debug_mode = _dbg
    torch.manual_seed(0x5EED)
    # SEEDED marker lets a silently-failed seed be caught; sbatch fails if absent from both parties.
    print(f"SEEDED|party={rank}|next=0x{0xDEADBEEF + rank:x}|"
          f"local=0x{0xC0FFEE + rank:x}|global=0x5eed|torch=0x5eed",
          file=sys.stderr, flush=True)

    # Write PID for the poller: spawned CrypTen parties have a generic cmdline with no script name,
    # so a cmdline filter cannot find them and the poller reads these pid files instead.
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

    # Communication transcript as integers (CrypTen's own 2-decimal print is only 5 MB resolution),
    # from the same run as the peak and gate; PROCEDURE.md phase B rests on exact byte counts.
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
        # Correctness gate observable: full-precision decrypted logits (the label alone is one bit
        # and would pass materially-different runs). Lives in the baseline, so applies to every step.
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
