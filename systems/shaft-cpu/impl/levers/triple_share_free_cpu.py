"""s2_triple_share_free: Beaver triple shares built without two whole-table scratch copies.
Basis: s1_plaintext_free peak 3059428 kB (second choice after s2_beaver_reveal_inplace measured nothing,
now off-path): int64 (28996,768) word-embedding share, 712605696 B / 0.233. Peak at after_triple, not
after_reveal (probe_embed_trace_cpu.py; the census misses the 5%-narrow spike).
Cost inside TrustedFirstParty.generate_additive_triple: ArithmeticSharedTensor(b, precision=0)
allocates an identity copy in encode (1 * x.long()) and a third PRZS buffer: five (28996,768) int64
tensors live (890757120 B) where three suffice. Fix: draw the PRZS pair, subtract in place, add the
source directly. Draws unchanged (calls, generator, order, size; values bit-identical).
Scope: generate_additive_triple only (PRZS and FixedPointEncoder.encode unchanged). Not taken
(off-path.md): skipping the non-source party's discarded a/b/c (advances its local generator differently).
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
    """PRZS with the subtraction in place: two buffers instead of three. Same two generators in the
    same order as ArithmeticSharedTensor.PRZS (bit-identical); sub_ reuses current_share."""
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
    """The stock ArithmeticSharedTensor(tensor, precision=0, src=src) minus the identity copy: encode
    at precision 0 is `1 * tensor.long()` == tensor but allocates, so add tensor directly. add_ mutates
    the PRZS buffer, never tensor."""
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
