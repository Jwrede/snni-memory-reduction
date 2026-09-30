"""DIAGNOSTIC ONLY. Adds SHAFT_INPUT_SEED_OFFSET to the torch seed and to nothing else, so the run draws a
DIFFERENT input sequence (the driver draws its token ids with torch.randint right after seeding torch)
while every CrypTen seed, and with it every mask and triple, stays the published one. It answers the
question behind the data-obliviousness argument for local memory: does the resident peak depend on the
input values? Refuses a zero offset (would reproduce the published input). Never in a measured step.
"""

import os
import sys

import torch

_OFF_RAW = os.environ.get("SHAFT_INPUT_SEED_OFFSET", "")
try:
    _OFF = int(_OFF_RAW, 0)
except ValueError:
    _OFF = 0

if _OFF == 0:
    print("LEVER|probe_input_seed_cpu|FATAL: SHAFT_INPUT_SEED_OFFSET missing or zero "
          f"(got {_OFF_RAW!r}); a control run that reproduces the published input is not a control",
          file=sys.stderr, flush=True)
    raise SystemExit(2)

_TORCH_ORIG = torch.manual_seed


def torch_manual_seed(seed):
    eff = seed + _OFF
    print(f"INPUTSEED|torch|offset=0x{_OFF:x}|seed=0x{eff:x}", file=sys.stderr, flush=True)
    return _TORCH_ORIG(eff)


torch.manual_seed = torch_manual_seed

print(f"LEVER|probe_input_seed_cpu|patched offset=0x{_OFF:x} scope=input", file=sys.stderr, flush=True)
