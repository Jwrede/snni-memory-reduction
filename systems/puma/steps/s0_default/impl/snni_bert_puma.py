# PUMA (SecretFlow SPU, ABY3 3PC pure MPC): BERT-base, 12 layers, sequence length 128.
#
# THIS FILE IS THE MEASURED PROGRAM. It is bind-mounted rather than baked into the image, exactly
# as SHAFT-CPU's driver is, so that a lever can be a module swapped at import time instead of a
# rebuild. Its md5 is written into the run's metadata, because the md5 of a bind-mounted script IS
# part of the identity of what was measured: change a line here and every later step is being gated
# against a baseline that no longer exists.
#
# It differs from the earlier campaign's flax_bert.py in exactly three ways, and each is part of
# the measured system, applied identically to the baseline and to every step:
#
#   1. SEEDS ARE PINNED where this stack lets them be pinned, and the run FAILS if a pin did not
#      take. See "What is pinned and what is not" below -- the honest answer is that SPU's share
#      randomness is not reachable from here, which is a finding rather than an omission.
#   2. IT EMITS A GATE OBSERVABLE. Without one there is nothing to compare a step against, and the
#      failure is silent, because the baseline defines the reference rather than being compared to
#      it. SHAFT-GPU ran an entire campaign that way.
#   3. IT EMITS MARKERS the harness checks: SEEDED|, WORKLOAD|, GATE|. A run whose lever silently
#      failed to load, or whose workload was not the one intended, otherwise looks exactly like a
#      successful measurement and publishes a plausible number.
#
# WHAT IS PINNED AND WHAT IS NOT.
#
# Pinned: numpy's global generator, JAX's PRNG key for the classifier head that
# `from_pretrained` initialises freshly, and SPU's `public_random_seed`. Together these fix
# everything on the PLAINTEXT side, so two runs feed the protocol identical inputs and identical
# weights.
#
# NOT pinned, and not pinnable from here: the secret shares. SPU draws them inside the runtime
# through `SecureRandSeed` (`common.cc`, `prg_state.cc`), which takes OS entropy and has no config
# knob; `RuntimeConfig.public_random_seed` is documented for PUBLIC variables only. This is the
# same stack, and the same limitation, that made BumbleBee "not bit-reproducible, reproducible
# within a mechanism-derived bound". So this system's tier is decided by the two-run check and
# recorded in SEEDS.md rather than assumed here, and the tolerance -- if one is needed -- is
# derived in MANIFEST.md from the protocol's truncation error before any pair of runs is compared.
# A tolerance fitted to the observed spread is not a tolerance.
#
# THE OBSERVABLE IS THE ENCODER OUTPUT, NOT THE CLASSIFIER LOGITS, and that is deliberate.
# `bert-base-uncased` ships no fine-tuned classification head, so `from_pretrained` initialises one
# at random: two logits computed from an untrained matrix. The encoder's last hidden state is what
# this workload actually computes -- 128 x 768 = 98,304 values, the same width as BumbleBee's
# observable -- and it is what a step must not change. The logits are emitted too, as evidence, not
# as the gate.
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
    # stderr, unbuffered, one line: the harness greps these and fails the run when one is missing.
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

# SPU's public randomness, pinned in the runtime config that the device is built from. It does NOT
# reach the secret shares -- see the header -- and is pinned anyway, because an unpinned knob that
# COULD have been pinned is indistinguishable, after the fact, from one that could not.
spu_cfg = conf["devices"]["SPU"]["config"]["runtime_config"]
spu_cfg["public_random_seed"] = SEED
if int(os.environ.get("PUMA_N_LAYERS_ASSERT", "1")):
    # A layer count other than 12 is a different program. It is allowed (a diagnostic may want
    # one layer, exactly as BumbleBee's does) but it is never silent.
    marker("WORKLOAD|model=%s|layers=%d|seq=%d|protocol=%s|field=%s"
           % (MODEL, N_LAYERS, SEQ, spu_cfg.get("protocol"), spu_cfg.get("field")))

ppd.init(conf["nodes"], conf["devices"])
marker("SEEDED|numpy+jax+spu_public|seed=0x%X" % SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
# seed= is the PRNG key `from_pretrained` uses for the freshly initialised classification head.
# Left to its default it is deterministic today and is pinned anyway, so that a transformers
# upgrade cannot change the measured program without changing this file.
pretrained_model = FlaxBertForSequenceClassification.from_pretrained(MODEL, seed=SEED & 0xFFFF)
params = pretrained_model.params
if N_LAYERS != 12:
    # Drop the trailing encoder layers rather than re-instantiating a smaller model: the remaining
    # weights are then bit-identical to the 12-layer run's, so a depth diagnostic differs from the
    # published run in depth alone.
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
    model = FlaxBertForSequenceClassification.from_pretrained(MODEL, seed=SEED & 0xFFFF)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        params=params,
        output_hidden_states=True,
    )
    # The encoder output is the gate; the logits ride along as evidence. Only these two leave the
    # SPU device: returning all 13 hidden states would put twelve more 98,304-value tensors into
    # the reconstruction and change the memory this campaign is measuring.
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
    # ONE VALUE PER LINE, full precision. The comparison can then be exact (T1) or numeric against
    # a declared tolerance (T2) without the file changing shape, which is what let BumbleBee's tier
    # be decided by measurement instead of by the format.
    #
    # Written to the file the harness names, and the harness fails the run if it is missing or
    # short: a run that produced no gate output is not a run with a passing gate. It goes to a file
    # rather than through stderr because stderr is shared with the runtime's own logging, and a
    # value-per-line observable spliced with foreign newlines silently lost four fifths of
    # BumbleBee's gate once.
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
    # A DEGENERATE OBSERVABLE IS NOT AN OBSERVABLE. SHARK's first gate was 98,304 exact zeros and
    # could not fail; SIGMA-GPU's was the same defect under a different cause. Checking that a
    # value EXISTS is not checking that it could have come out differently.
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

    # The plaintext run is a REFERENCE, never the gate: it tests whether the system is correct,
    # while the gate tests whether this step changed anything, and only the second is what a
    # waterfall row claims. It is published because the distance between them is the honest
    # measure of what the protocol's fixed-point arithmetic costs at this depth.
    d = np.abs(np.asarray(spu_hidden, dtype=np.float64)
               - np.asarray(cpu_hidden, dtype=np.float64))
    ref = np.abs(np.asarray(cpu_hidden, dtype=np.float64))
    marker("REFERENCE|spu_vs_cpu|max_abs=%.6g|max_rel=%.6g|median_magnitude=%.6g"
           % (float(d.max()), float((d / np.maximum(ref, 1e-12)).max()), float(np.median(ref))))
    marker("MARK|done|t=%.1f" % (time.time() - t0))
    sys.exit(0 if ok else 6)
