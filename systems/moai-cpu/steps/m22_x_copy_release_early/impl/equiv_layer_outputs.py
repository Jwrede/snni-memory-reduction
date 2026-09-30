#!/usr/bin/env python3
"""Phase-B plaintext test for MOAI-CPU: the step computes the baseline's function.

    equiv_layer_outputs.py <step-dir> <baseline-step-dir>

Single non-interactive process: transcript condition vacuous (MANIFEST transcript_none); equivalence
rests on this test. Compared: decrypted layer outputs (layer_outputs/layer_<id>.txt, flattened into
gate.txt, 7,680 values over two layers, |x| up to 11.4). Repartitioning the bootstrap (phase split,
chunking) or the FFN (tiling) re-draws CKKS bootstrapping error; outputs move at that level.
Bound: four bootstraps at scale 2^46, degree-59 sine, 2^-9-level modular reduction leave ~8 to 9
correct bits in the worst slots: 2^-8 absolute (3.9e-3) per value; sign fixed for values above 2^-6.
Characterisation of the scheme's precision, not pinned before the line was measured (planned as
phase A; first phase-split port refuted as a bug, off-path.md).
"""
import math
import os
import sys


def load(step_dir):
    f = os.path.join(step_dir, "logs", "run1", "gate.txt")
    if not os.path.exists(f):
        f = os.path.join(step_dir, "gate.txt")
    return [float(l.split()[0]) for l in open(f) if l.strip()]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[2]), load(sys.argv[1])
    if len(a) != len(b) or not a:
        sys.exit(f"FAIL: {len(b)} values against the baseline's {len(a)}")
    bound = 2.0 ** -8
    diff = [abs(x - y) for x, y in zip(a, b)]
    mx = max(diff)
    rms_a = math.sqrt(sum(x * x for x in a) / len(a))
    rms_d = math.sqrt(sum(d * d for d in diff) / len(diff))
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    var = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    corr = cov / var if var else 0.0
    signs = sum(1 for x, y in zip(a, b) if abs(x) > 2.0 ** -6 and (x > 0) != (y > 0))
    within = sum(1 for d in diff if d <= bound)
    print(f"values {len(a)}  |x| max {max(abs(x) for x in a):.4g}  rms {rms_a:.4g}")
    print(f"max |diff| {mx:.3e}  rms diff {rms_d:.3e}  rel rms {rms_d / rms_a:.3e}  corr {corr:.9f}")
    print(f"within 2^-8: {within}/{len(a)}  sign flips above 2^-6: {signs}")
    ok = mx <= bound and signs == 0
    print("EQUIVALENT: yes" if ok else "EQUIVALENT: NO")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
