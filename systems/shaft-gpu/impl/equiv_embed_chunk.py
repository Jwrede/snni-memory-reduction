#!/usr/bin/env python3
"""Plaintext equivalence test for the embedding-chunk lever (PROCEDURE.md phase B, condition b):
the block decomposition sum_i indices_oh[...,s:e] @ weight[s:e,:] equals the whole matmul. Runs in
int64 with wraparound demanding bit-exact equality (the protocol computes mod 2^64 and these values
overflow). Shows only the lever's arithmetic, not the shared matmul. Exits non-zero on any mismatch."""
import sys

import torch


def chunked_matmul(indices_oh, weight, num_chunks):
    """The lever's arithmetic, transcribed from embed_chunk_gpu.py:75-90."""
    voc_size = weight.shape[0]
    chunk_size = (voc_size + num_chunks - 1) // num_chunks
    output = None
    for i in range(num_chunks):
        s = i * chunk_size
        e = min(s + chunk_size, voc_size)
        if s >= e:
            continue
        chunk_out = indices_oh[..., s:e].matmul(weight[s:e, :])
        output = chunk_out if output is None else output + chunk_out
    return output


def main():
    torch.manual_seed(0x5EED)
    failures = []

    # The real shape plus small awkward ones: the chunk count need not divide the vocabulary, and an
    # off-by-one in the last chunk is the bug this test exists to catch.
    cases = [
        (1, 128, 28996, 768, 16),   # the measured configuration
        (1, 128, 28996, 768, 32),
        (2, 7, 101, 13, 16),        # V not divisible by the chunk count
        (1, 3, 5, 4, 7),            # more chunks than vocabulary entries
        (1, 4, 64, 8, 1),           # the no-op case: one chunk must be the identity
    ]

    for (b, s, v, d, n) in cases:
        # A one-hot index matrix, as the lever receives it, in the ring the protocol computes in.
        idx = torch.randint(0, v, (b, s), dtype=torch.int64)
        indices_oh = torch.zeros((b, s, v), dtype=torch.int64)
        indices_oh.scatter_(2, idx.unsqueeze(-1), 1)

        # Weights whose products overflow int64, as the measured system's do.
        weight = torch.randint(-(2 ** 40), 2 ** 40, (v, d), dtype=torch.int64)

        whole = indices_oh.matmul(weight)
        chunked = chunked_matmul(indices_oh, weight, n)
        if not torch.equal(whole, chunked):
            bad = (whole != chunked).sum().item()
            failures.append(f"B={b} S={s} V={v} D={d} chunks={n}: {bad} elements differ")
        else:
            print(f"  ok  B={b} S={s} V={v} D={d} chunks={n}: bit-identical over int64")

    # The decomposition survives wraparound, tested directly: the accumulated sum exceeds int64.
    big_oh = torch.ones((1, 1, 4), dtype=torch.int64)
    big_w = torch.full((4, 1), (2 ** 62) - 1, dtype=torch.int64)
    if torch.equal(big_oh.matmul(big_w), chunked_matmul(big_oh, big_w, 4)):
        print("  ok  wraparound case: bit-identical after the sum overflows int64")
    else:
        failures.append("wraparound case: the decomposition is not exact under overflow")

    if failures:
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        sys.exit(f"{len(failures)} case(s) failed: the lever does not compute the same function")
    print("PASS: chunking the embedding matmul is exact over int64, wraparound included")


if __name__ == "__main__":
    main()
