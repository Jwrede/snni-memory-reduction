"""s2_triple_share_free: build a Beaver triple's shares without two whole-table scratch copies.

CHOSEN FROM THE MEASURED DECOMPOSITION OF s1_plaintext_free's PEAK (3059428 kB), not inherited, and
chosen a SECOND time after the first attempt on the same object measured nothing. The object did not
change; the identification of where its copies came from did. PROCEDURE.md's rule applies: a failed
attempt is evidence about the attempt, not about the object, so the largest object stays the target
until its reducibility is disproven.

THE OBJECT, unchanged from s1's ranking: `int64 (28996, 768)`, the secret share of the word-embedding
table, 712605696 B at s1's peak and 0.233 of it, the largest object there.

WHAT THE FIRST ATTEMPT GOT WRONG. `s2_beaver_reveal_inplace` (now in `off-path/`) removed two
genuinely redundant copies -- the clone inside `y - b` and the clone inside `all_reduce` -- and moved
the peak by 1.15%, inside its own 1.171% spread. The reason is measured, not guessed:
`impl/levers/probe_embed_trace_cpu.py` brackets RSS and `VmHWM` at every allocation boundary of the
embedding's Beaver matmul, and `VmHWM` reaches its FINAL value at `after_triple`:

    EMBED|entry           rss 2002224  hwm 2674028  live n=1
    EMBED|after_triple    rss 2321520  hwm 3009608  live n=2     <- the peak happens in here
    EMBED|after_y_minus_b rss 2553492  hwm 3009608  live n=3
    EMBED|after_reveal    rss 2760456  hwm 3009608  live n=4

The reveal is where the census saw four live copies, and it is NOT where the peak is. The census
samples on the way up and never caught the spike, which is the same 5%-narrow spike the measured runs
report as the gap between the polled maximum and `VmHWM`.

WHERE THE PEAK ACTUALLY IS. Inside `TrustedFirstParty.generate_additive_triple`
(`mpc/provider/tfp_provider.py:24`), `b` is a (28996, 768) int64 random tensor and is then wrapped by
`ArithmeticSharedTensor(b, precision=0, src=0)`. That constructor does two things that each allocate a
full extra copy of the table and read it once:

  1. `tensor = self.encoder.encode(tensor)` (`arithmetic.py:91`). At `precision=0` the scale is 1 and
     the int-tensor branch is `return self._scale * x.long()` (`encoder.py:60`). `x.long()` on an
     int64 tensor is the same tensor, and `1 * x` is an identity multiply that ALLOCATES A NEW ONE.

  2. `ArithmeticSharedTensor.PRZS` (`arithmetic.py:158`) draws two random tensors and returns
     `current_share - next_share`, so three buffers are live where two suffice.

At the instant of that subtraction the live (28996, 768) int64 tensors are: the raw `b`, the identity
copy from `encode`, `current_share`, `next_share`, and the difference -- FIVE, 890757120 B. Three are
enough.

THE FIX, and why it is free. `generate_additive_triple` is reimplemented to draw the PRZS pair and
subtract IN PLACE, and to add the source tensor directly rather than an identity copy of it. Nothing
is recomputed and no communication happens here at all: this function is local.

THE DRAWS ARE UNTOUCHED, which on this system is the binding constraint. Fixed-point truncation is
share-dependent, so any change to the random draws moves the logits far past the 1e-6 tolerance --
that is what the unseeded baseline's 3.8x spread was. Every draw here is the same call, on the same
generator, in the same order, with the same size: `local` for `a` and `b`, then `prev` and `next` for
each PRZS. The values produced are bit-identical; only the buffers differ.

SCOPE. Only `TrustedFirstParty.generate_additive_triple` is replaced. `PRZS` and
`FixedPointEncoder.encode` are left alone, because both are used elsewhere on tensors that are still
live afterwards, where an in-place subtraction or a shared buffer would corrupt the caller's data.

NOT AN OFF-PATH TRADE: no accuracy, no security property and no portability is given up.

WHAT IS DELIBERATELY NOT TOUCHED, and why it is in off-path.md rather than here: both parties compute
`a`, `b` and `c` from their own `local` generator and the non-source party then discards its copies.
Skipping that work on the non-source party would save more than this lever does, and it would advance
that party's `local` generator differently, which changes every subsequent share. It is not free.
"""

import sys

import torch

import crypten
import crypten.communicator as comm
from crypten.common.rng import generate_random_ring_element
from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
from crypten.mpc.provider.tfp_provider import TrustedFirstParty

_ORIG = TrustedFirstParty.generate_additive_triple


def _przs_share_inplace(size, device):
    """PRZS, with the subtraction done in place: two buffers instead of three.

    Same two generators in the same order as `ArithmeticSharedTensor.PRZS`, so the numbers drawn and
    the value returned are bit-identical. `sub_` reuses `current_share`, which has no other referrer.
    """
    from crypten import generators

    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    g0 = generators["prev"][device]
    g1 = generators["next"][device]
    current_share = generate_random_ring_element(size, generator=g0, device=device)
    next_share = generate_random_ring_element(size, generator=g1, device=device)
    return current_share.sub_(next_share)


def _share_of(tensor, size, device, src=0):
    """The stock `ArithmeticSharedTensor(tensor, precision=0, src=src)`, minus the identity copy.

    The stock path is `share = PRZS(size); if rank == src: share += encode(tensor)`. At precision 0
    `encode` returns `1 * tensor.long()`, which equals `tensor` and allocates. Adding `tensor`
    directly is the same arithmetic. `add_` mutates the PRZS buffer, never `tensor`.
    """
    share = _przs_share_inplace(size, device)
    if comm.get().get_rank() == src:
        share.add_(tensor)
    return ArithmeticSharedTensor.from_shares(share, precision=0)


def generate_additive_triple(self, size0, size1, op, device=None, *args, **kwargs):
    a = generate_random_ring_element(size0, device=device)
    b = generate_random_ring_element(size1, device=device)
    c = getattr(torch, op)(a, b, *args, **kwargs)
    csize = c.size()

    a_s = _share_of(a, size0, device)
    del a
    b_s = _share_of(b, size1, device)
    del b
    c_s = _share_of(c, csize, device)
    del c
    return a_s, b_s, c_s


TrustedFirstParty.generate_additive_triple = generate_additive_triple

print("LEVER|triple_share_free_cpu|patched", file=sys.stderr, flush=True)
