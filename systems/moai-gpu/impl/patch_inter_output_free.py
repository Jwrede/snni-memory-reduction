#!/usr/bin/env python3
"""MOAI-GPU s3: release each inter_output element (test_single_layer.cuh:1073) when its gelu_v2 call
returns, instead of after the loop (co-resident with gelu_output at the after_gelu peak,
99,634.1 MiB).

    patch_inter_output_free.py <moai source tree>

Free (dead-value release). Loop omp parallel over i, static schedule, one writer per index. Gate T1s.
"""
import os
import re
import sys

SRC = "src/include/test/test_single_layer.cuh"

ANCHOR = """                gelu_output[i*32+j] = gelu_v2(inter_output[i*32+j],context,relin_keys,secret_key, stream);
"""

PATCH = """                gelu_output[i*32+j] = gelu_v2(inter_output[i*32+j],context,relin_keys,secret_key, stream);
                // s3_inter_output_free: this element is dead the instant gelu_v2 has read it, but
                // the stock code frees the whole 3072-ciphertext array only after the loop, so it
                // stays co-resident with gelu_output while that one grows to full size -- and
                // after_gelu is this system's highest measured pool occupancy. Index i*32+j is
                // touched by exactly one thread under the static schedule, so releasing it here
                // cannot race a reader.
                inter_output[i*32+j] = PhantomCiphertext();
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "s3_inter_output_free" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the gelu call matched {txt.count(ANCHOR)} times, expected exactly 1")

    # The loop must be the parallel one with a single writer per index (the safety argument);
    # without `omp parallel for` over i, a release here could race a read.
    at = txt.index(ANCHOR)
    head = txt[:at]
    if "#pragma omp parallel for" not in head[-800:]:
        sys.exit("FAILED: no `#pragma omp parallel for` above the gelu call; the single-writer "
                 "argument this step rests on cannot be made")

    # Nothing may read inter_output after this loop; comments stripped first so this step's own
    # comment cannot satisfy the check.
    tail = re.sub(r"//[^\n]*", "", txt[at + len(ANCHOR):])
    uses = re.findall(r"inter_output\s*\[", tail)
    if uses:
        sys.exit(f"FAILED: `inter_output[...]` is read {len(uses)} more time(s) after the gelu "
                 "loop; releasing it there would lose something still in use")

    txt = txt.replace(ANCHOR, PATCH, 1)
    open(src, "w").write(txt)
    print("patch_inter_output_free: inter_output released as GELU consumes it (" + SRC + ")")


if __name__ == "__main__":
    main()
