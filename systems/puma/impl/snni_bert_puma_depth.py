"""DIAGNOSTIC DRIVER, generated from snni_bert_puma.py 2026-08-04: passes num_hidden_layers=N_LAYERS
to both from_pretrained calls so config and params agree below 12 layers. Separate file so the
measured driver's md5 (its identity in RUN_META) stays untouched.
"""
# PUMA (SecretFlow SPU, ABY3 3PC pure MPC): BERT-base, 12 layers, sequence length 128.
#
# This file is the measured program, bind-mounted (not baked) so a lever is a module swap; its md5
# is part of the run identity. It differs from the earlier flax_bert.py by pinning seeds (run fails
# if a pin did not take), emitting a gate observable, and emitting SEEDED|/WORKLOAD|/GATE| markers.
# Pinned: numpy, JAX classifier-head key, SPU public_random_seed (the plaintext side). Not pinnable:
# the secret shares (SecureRandSeed, OS entropy), so the tier is decided by the two-run check in
# SEEDS.md. The gate observable is the encoder's last hidden state (98,304 values), not the random
# classifier logits, which ride along as evidence.
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

MODEL = "bert-base-uncased"
SEQ = int(os.environ.get("PUMA_SEQ_LEN", "128"))
N_LAYERS = int(os.environ.get("PUMA_N_LAYERS", "12"))
PROMPT = "I enjoy walking with my cute dog"
SEED = int(os.environ.get("SNNI_SEED", "0x5EED"), 0)


def marker(text):
    # stderr, unbuffered: the harness greps these and fails the run when one is missing.
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


parser = argparse.ArgumentParser(description="SNNI measured driver (PUMA, BERT-base).")
parser.add_argument("-c", "--config", default="/opt/puma/examples/python/ml/flax_bert/3pc.json")
args = parser.parse_args()

marker("SNNI_SCRIPT|md5=%s" % hashlib.md5(open(__file__, "rb").read()).hexdigest())

# ---------------------------------------------------------------------------------------------
# Seeds, before anything draws.
np.random.seed(SEED & 0xFFFFFFFF)
os.environ.setdefault("PYTHONHASHSEED", str(SEED & 0xFFFFFFFF))

import jax  # noqa: E402  (imported after the seed is set, so nothing draws first)
import jax.numpy as jnp  # noqa: E402
from transformers import AutoTokenizer, FlaxBertForSequenceClassification  # noqa: E402

import examples.python.utils.distributed as ppd  # noqa: E402

with open(args.config, "r") as f:
    conf = json.load(f)

# SPU public randomness, pinned in the runtime config; does not reach the secret shares.
spu_cfg = conf["devices"]["SPU"]["config"]["runtime_config"]
spu_cfg["public_random_seed"] = SEED
if int(os.environ.get("PUMA_N_LAYERS_ASSERT", "1")):
    # A layer count other than 12 is a different program: allowed for diagnostics, never silent.
    marker("WORKLOAD|model=%s|layers=%d|seq=%d|protocol=%s|field=%s"
           % (MODEL, N_LAYERS, SEQ, spu_cfg.get("protocol"), spu_cfg.get("field")))

ppd.init(conf["nodes"], conf["devices"])
marker("SEEDED|numpy+jax+spu_public|seed=0x%X" % SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
# seed= pins the PRNG key for the freshly initialised classification head.
pretrained_model = FlaxBertForSequenceClassification.from_pretrained(
    MODEL, seed=SEED & 0xFFFF, num_hidden_layers=N_LAYERS)
params = pretrained_model.params
if N_LAYERS != 12:
    # Drop trailing encoder layers (not a smaller model) so remaining weights are bit-identical.
    enc = params["bert"]["encoder"]["layer"]
    params = {**params}
    params["bert"] = {**params["bert"]}
    params["bert"]["encoder"] = {"layer": {str(i): enc[str(i)] for i in range(N_LAYERS)}}


def _encode():
    return tokenizer(
        PROMPT, return_tensors="jax",
        padding="max_length", max_length=SEQ, truncation=True,
    )


def bert_forward(input_ids, attention_mask, token_type_ids, params):
    model = FlaxBertForSequenceClassification.from_pretrained(
        MODEL, seed=SEED & 0xFFFF, num_hidden_layers=N_LAYERS)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        params=params,
        output_hidden_states=True,
    )
    # Encoder output is the gate, logits ride as evidence; only these two leave the SPU device.
    return out.logits, out.hidden_states[-1]


def run_on_cpu():
    enc = _encode()
    return bert_forward(enc["input_ids"], enc["attention_mask"],
                        enc["token_type_ids"], params)


def run_on_spu():
    enc = _encode()
    input_ids = ppd.device("P1")(lambda x: x)(enc["input_ids"])
    attention_mask = ppd.device("P1")(lambda x: x)(enc["attention_mask"])
    token_type_ids = ppd.device("P1")(lambda x: x)(enc["token_type_ids"])
    p = ppd.device("P2")(lambda x: x)(params)
    logits, hidden = ppd.device("SPU")(bert_forward)(
        input_ids, attention_mask, token_type_ids, p
    )
    return ppd.get(logits), ppd.get(hidden)


def write_gate(hidden, logits):
    # One value per line, full precision, to a named file (not stderr, which is shared with runtime
    # logging); the harness fails the run if it is missing or short.
    path = os.environ.get("PUMA_GATE_FILE")
    flat = np.asarray(hidden, dtype=np.float64).reshape(-1)
    if path:
        with open(path, "w") as fh:
            for v in flat:
                fh.write("%.17g\n" % v)
    marker("GATE|hidden_last|n=%d|sum=%.17g|absmax=%.17g"
           % (flat.size, float(flat.sum()), float(np.abs(flat).max())))
    lg = np.asarray(logits, dtype=np.float64).reshape(-1)
    marker("GATE|logits|" + "|".join("%.17g" % v for v in lg))
    # A degenerate (all-zero) observable cannot fail, so check for nonzero values.
    nz = int(np.count_nonzero(flat))
    marker("GATE_NONZERO|%d/%d" % (nz, flat.size))
    if nz == 0:
        marker("GATE_DEGENERATE|every value is zero")
        return False
    return True


if __name__ == "__main__":
    t0 = time.time()
    cpu_logits, cpu_hidden = run_on_cpu()
    marker("REFERENCE|cpu_logits|" + "|".join(
        "%.17g" % v for v in np.asarray(cpu_logits, dtype=np.float64).reshape(-1)))
    marker("MARK|cpu_reference_done|t=%.1f" % (time.time() - t0))

    spu_logits, spu_hidden = run_on_spu()
    marker("MARK|spu_inference_done|t=%.1f" % (time.time() - t0))

    ok = write_gate(spu_hidden, spu_logits)

    # The plaintext run is a reference (is the system correct), never the gate (did this step
    # change anything); its distance from SPU measures the fixed-point cost.
    d = np.abs(np.asarray(spu_hidden, dtype=np.float64)
               - np.asarray(cpu_hidden, dtype=np.float64))
    ref = np.abs(np.asarray(cpu_hidden, dtype=np.float64))
    marker("REFERENCE|spu_vs_cpu|max_abs=%.6g|max_rel=%.6g|median_magnitude=%.6g"
           % (float(d.max()), float((d / np.maximum(ref, 1e-12)).max()), float(np.median(ref))))
    marker("MARK|done|t=%.1f" % (time.time() - t0))
    sys.exit(0 if ok else 6)
