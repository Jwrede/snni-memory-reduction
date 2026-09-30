"""s2_beaver_reveal_inplace: no throwaway copies of the masked operand.
At s1's peak (3059428 kB) the int64 (28996,768) embedding share exists four times; two are bookkeeping
(ArithmeticSharedTensor._arithmetic_function's clone for y-b; all_reduce's input clone). x-a and y-b
built without the clone (torch.sub on shares) and reduced in place. Nothing recomputed, no round
added, wire bytes and draws unchanged. all_reduce patched behind a flag set only here; the replacement
is named all_reduce, _logging-decorated, batched=True (comm bytes/rounds reported identically).
"""

import sys

import torch
import torch.distributed as dist

import crypten
from crypten.communicator.communicator import _logging
from crypten.communicator.distributed_communicator import DistributedCommunicator
from crypten.mpc.primitives import beaver

_INPLACE = False
_ORIG_ALL_REDUCE = DistributedCommunicator.all_reduce
# The undecorated function (via __wrapped__), so delegating to it does not log the comm twice.
_RAW_ALL_REDUCE = getattr(_ORIG_ALL_REDUCE, "__wrapped__", None)
if _RAW_ALL_REDUCE is None:  # pragma: no cover - would mean CrypTen stopped using functools.wraps
    raise RuntimeError("cannot reach the undecorated all_reduce; refusing to double-count comm")


@_logging
def all_reduce(self, input, op=dist.ReduceOp.SUM, batched=False):
    if batched and _INPLACE:
        reqs = [
            dist.all_reduce(t.data, op=op, group=self.main_group, async_op=True) for t in input
        ]
        for req in reqs:
            req.wait()
        return input
    return _RAW_ALL_REDUCE(self, input, op=op, batched=batched)


def _beaver_matmul(x, y, *args, **kwargs):
    global _INPLACE
    provider = crypten.mpc.get_default_provider()
    # Same call, sizes and order, so the generators are consumed identically (same masks).
    a, b, c = provider.generate_additive_triple(
        x.size(), y.size(), "matmul", device=x.device, *args, **kwargs
    )

    # torch.sub on the shares rather than x - a: one allocation instead of a clone plus one, and no
    # encoding (which is all IgnoreEncodings suppressed), so the same arithmetic.
    e_share = torch.sub(x.share, a.share)
    d_share = torch.sub(y.share, b.share)

    _INPLACE = True
    try:
        epsilon, delta = crypten.communicator.get().all_reduce(
            [e_share, d_share], batched=True
        )
    finally:
        _INPLACE = False

    # Stock accumulation: the epsilon*delta term is PUBLIC, added by rank 0 only, which
    # c += <torch tensor> handles and c._tensor += would not.
    c._tensor += torch.matmul(epsilon, b._tensor, *args, **kwargs)
    c._tensor += torch.matmul(a._tensor, delta, *args, **kwargs)
    c += torch.matmul(epsilon, delta, *args, **kwargs)
    return c


DistributedCommunicator.all_reduce = all_reduce
beaver.matmul = _beaver_matmul

print("LEVER|beaver_reveal_inplace_cpu|patched", file=sys.stderr, flush=True)
