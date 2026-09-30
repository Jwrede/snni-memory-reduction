#!/usr/bin/env python3
"""Plaintext equivalence test for chunked parameter encryption (PROCEDURE.md phase B, condition (b)).
encrypt_chunk_cpu replaces one cryptensor() with crypten.cat over torch.chunk(N). Checks: split
bit-identical on the actual shapes; encrypted path within one 16-bit fixed-point ULP (1/65536) of the
unchunked; no drift with N (2, 4, 8, 16). Threshold selection reported separately
(LEVER|encrypt_chunk_cpu|chunked_params=<n>).
Run inside the measured image: python3 equiv_encrypt_chunk.py (exits non-zero on any mismatch).
"""

import sys

import torch

import crypten


# Shapes the lever sees, largest first. Only the first is above the 5,000,000-element threshold;
# the others cover the case the lever must NOT touch.
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
    # Embedding shape at reduced rows: the equivalence is a property of the split, not the size, and
    # encrypting 22.3M elements four times over is minutes of protocol for what 4096 rows answer.
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
