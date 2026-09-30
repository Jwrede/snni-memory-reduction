"""DIAGNOSTIC ONLY. Shifts every seed by a constant, so the gate observable can be measured under
DIFFERENT randomness while nothing else about the program changes.

WHY THIS EXISTS. The gate is declared T2 with a 1e-6 relative tolerance, but the observable is
CrypTen fixed point with 16 fractional bits: the published logits `1.3482666016e-01` and
`-2.3132324219e-01` are exactly 8836/65536 and -15160/65536, so one ulp is 2^-16 = 1.526e-5
absolute, which on the first component is 1.1e-4 relative -- 113 times the tolerance. A tolerance
set two orders of magnitude below the resolution of the thing it measures can only ever admit
byte-identical output, so T2 at 1e-6 is T1 wearing a tolerance.

That matters because fixed-point truncation in this protocol is SHARE-dependent: with two parties
`div_` truncates each share locally and uncorrected (`mpc/primitives/arithmetic.py:462-469`), so
the last bits of the reconstruction depend on whether the shares wrapped the ring, and that depends
on the masks. Any lever that changes which random values are drawn -- chunking a matmul,
repartitioning a loop, encrypting a layer at its first use rather than up front -- moves the output
by at least an ulp and fails the gate, whatever it does to memory and however correct it is. The
retired campaign's largest SHAFT lever is exactly of that kind and its line had no gate at all.

So the question this probe answers is empirical and it decides which levers are admissible:
**how far does the observable move when ONLY the randomness changes?** That number is the
protocol's own mask-induced spread. A tolerance measured from it accepts a step that drew different
masks and still rejects a step that changed the computation, which is what a gate is for; a chosen
tolerance below one ulp cannot tell those two apart. It is the same move the campaign already makes
for `cost_type`, which compares a runtime delta against that step's OWN measured spread rather than
against a threshold somebody picked.

WHAT IT DOES. `SHAFT_SEED_OFFSET` is added to every seed the measured script passes, leaving the
per-rank offsets intact (so the parties still hold independent generator state) and leaving the
global seed shared between them (so it is still one protocol). Nothing else is touched: same image,
same code path, same thread count, same input.

REFUSES A ZERO OFFSET, because a sweep run that silently reproduced the published seed would look
exactly like a run that had swept and found no movement -- the failure shape this campaign keeps
meeting.

NEVER IN A MEASURED RUN. `SHAFT_LEVERS` names it explicitly and RUN_META records that, so a step
whose RUN_META names this module is not a measurement. Note that the measured script's own
`SEEDED|` line prints the values it PASSED, not the shifted ones; this module prints the effective
seeds separately, and the two differing is the evidence that the shift took.
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
