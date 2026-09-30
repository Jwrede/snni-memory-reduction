"""Diagnostic only. Adds SHAFT_SEED_OFFSET to every seed: gate observable under different randomness,
nothing else changed (protocol's mask-induced spread = gate tolerance; 1e-6 is below one 16-bit
fixed-point ulp, 1.526e-5). Fixed-point truncation is share-dependent (div_ truncates each share
locally). Per-rank offsets kept, global seed shared. Zero offset refused. Never in a measured run; the
measured script's SEEDED| line prints passed values, this module the effective seeds.
"""

import os
import sys

import torch

import crypten as ct

_OFF_RAW = os.environ.get("SHAFT_SEED_OFFSET", "")
try:
    _OFF = int(_OFF_RAW, 0)
except ValueError:
    _OFF = 0

if _OFF == 0:
    print("LEVER|probe_seed_sweep_cpu|FATAL: SHAFT_SEED_OFFSET missing or zero "
          f"(got {_OFF_RAW!r}); a sweep run that reproduces the published seed is not a sweep",
          file=sys.stderr, flush=True)
    raise SystemExit(2)

_CT_ORIG = ct.manual_seed
_TORCH_ORIG = torch.manual_seed


def manual_seed(next_seed, local_seed, global_seed):
    eff = (next_seed + _OFF, local_seed + _OFF, global_seed + _OFF)
    print(f"SEEDSWEEP|crypten|offset=0x{_OFF:x}|next=0x{eff[0]:x}|local=0x{eff[1]:x}|"
          f"global=0x{eff[2]:x}", file=sys.stderr, flush=True)
    return _CT_ORIG(*eff)


def torch_manual_seed(seed):
    eff = seed + _OFF
    print(f"SEEDSWEEP|torch|offset=0x{_OFF:x}|seed=0x{eff:x}", file=sys.stderr, flush=True)
    return _TORCH_ORIG(eff)


# SHAFT_SEED_SCOPE=masks shifts only the CrypTen seeds (the masks and triples) and leaves the torch
# seed, and with it the random input sequence, unchanged, so a moved output is attributable to the
# masks alone. Unset or "all", every seed is shifted, which also draws a different input.
_SCOPE = os.environ.get("SHAFT_SEED_SCOPE", "all")
if _SCOPE not in ("all", "masks"):
    print(f"LEVER|probe_seed_sweep_cpu|FATAL: SHAFT_SEED_SCOPE={_SCOPE!r} is neither all nor masks",
          file=sys.stderr, flush=True)
    raise SystemExit(2)

ct.manual_seed = manual_seed
if _SCOPE == "all":
    torch.manual_seed = torch_manual_seed

print(f"LEVER|probe_seed_sweep_cpu|patched offset=0x{_OFF:x} scope={_SCOPE}", file=sys.stderr, flush=True)
