#!/usr/bin/env python3
"""Plaintext equivalence test, chunked parameter encryption. PROCEDURE.md phase B, condition (b).

encrypt_chunk_cpu replaces

    enc = crypten.cryptensor(param, src=src, requires_grad=rg)

with

    enc = crypten.cat([crypten.cryptensor(c.contiguous(), src=src, requires_grad=rg)
                       for c in torch.chunk(param, N, dim=0)], dim=0)

Eight broadcasts draw the mask stream in eight pieces; the exact gate cannot certify it. The transcript
covers protocol work, this test covers the function.

Tests:
  1. Split: torch.cat(torch.chunk(x, N, dim=0)) == x, bit-identical, on the actual shapes (row reordering
     would give plausible logits).
  2. Encrypted path: chunked vs unchunked decryption differ by less than one unit of the 16-bit
     fixed-point scale (1/65536 = 1.526e-05).
  3. Chunk count: comparison at N = 2, 4, 8, 16.
Not shown: that only the embedding crosses the threshold (reported by the run as
LEVER|encrypt_chunk_cpu|chunked_params=<n>).

Run inside the measured image: python3 equiv_encrypt_chunk.py   (non-zero exit on mismatch)
"""

import sys

import torch

import crypten


# The shapes this lever actually sees in the measured workload, largest first. Only the first is
# above the 5,000,000-element threshold; the others are here so the test also covers the case the
# lever must NOT touch.
SHAPES = [
    (28996, 768),    # the embedding table, 22.3M elements, the one that gets chunked
    (3072, 768),     # an FFN weight, 2.36M, below threshold
    (768, 768),      # an attention weight, 590k, below threshold
]

# One unit in the last place of CrypTen's default 16-bit fixed-point scale.
ULP = 1.0 / 65536


def main():
    crypten.init()
    torch.manual_seed(0x5EED)

    failures = []

    # ---- (1) the split, on plaintext, bit-identical ----
    for shape in SHAPES:
        x = torch.randn(*shape)
        for n in (2, 4, 8, 16):
            rebuilt = torch.cat(torch.chunk(x, n, dim=0), dim=0)
            if rebuilt.shape != x.shape:
                failures.append(f"chunk/cat changed shape at N={n}, {shape}: "
                                f"{tuple(rebuilt.shape)} != {tuple(x.shape)}")
            elif not torch.equal(rebuilt, x):
                failures.append(f"chunk/cat is not bit-identical at N={n}, {shape}")
    if not failures:
        print(f"  ok  chunk/cat is bit-identical for {len(SHAPES)} shapes at N = 2, 4, 8, 16")

    # ---- (2) and (3) the encrypted path, at four chunk counts ----
    # The embedding shape is used at a reduced number of rows: the equivalence is a property of the
    # split, not of the size, and encrypting 22.3M elements four times over is minutes of protocol
    # for a question that 4096 rows answer identically. The reduction is stated rather than silent.
    probe = (4096, 768)
    x = torch.randn(*probe)

    whole = crypten.cryptensor(x, src=0).get_plain_text()

    for n in (2, 4, 8, 16):
        parts = [crypten.cryptensor(c.contiguous(), src=0) for c in torch.chunk(x, n, dim=0)]
        chunked = crypten.cat(parts, dim=0).get_plain_text()

        if chunked.shape != whole.shape:
            failures.append(f"N={n}: shape {tuple(chunked.shape)} != {tuple(whole.shape)}")
            continue
        d = (chunked - whole).abs().max().item()
        if d >= ULP:
            failures.append(f"N={n}: max abs difference {d:.6e} is not below one fixed-point unit "
                            f"({ULP:.6e}); the chunked form is not encrypting the same values")
        else:
            print(f"  ok  N={n:2d}: chunked encryption agrees with whole to {d:.3e}, "
                  f"under one fixed-point unit {ULP:.3e}")

    if failures:
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        sys.exit(f"{len(failures)} case(s) failed: chunked encryption does not compute the same "
                 f"function as whole encryption")
    print("PASS: chunked parameter encryption computes the same function, at every chunk count "
          "tested, and the split itself is bit-identical")


if __name__ == "__main__":
    main()
