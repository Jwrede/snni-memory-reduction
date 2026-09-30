#!/usr/bin/env python3
"""MOAI-GPU s3: input copy taken at full level then shrunk; the two loops fused.

    patch_input_copy_fused.py <moai source tree>

Basis: pool growth events (p_finalprobe: peak_vram = reserved high-water). before_attention ratio
1.95: enc_ecd_x and its full-level copy both fresh (55,296 MiB demanded, 53,792 reserved, 27,648
held at phase end). Mod-switch moved into the copy loop (element i shrunk before i+1 is copied):
copy grows to 16,128 MiB instead of 27,648. Free (ordering; same operations and operands; elements
independent). Expected ~-26,000 MiB on 123,232 MiB (~-21%). Gate T1s.
"""
import os
import sys

SRC = "src/include/test/test_single_layer.cuh"

ANCHOR = """    vector<PhantomCiphertext> enc_ecd_x_copy(num_col);
    for (int i = 0; i < num_col; ++i){
        enc_ecd_x_copy[i] = enc_ecd_x[i];
    }

    // #pragma omp parallel for

    for (int i = 0; i < num_col; ++i) {
        for (int j = 0; j < boot_level; ++j){
            evaluator.mod_switch_to_next_inplace(enc_ecd_x_copy[i]);
        }
    }
"""

PATCH = """    vector<PhantomCiphertext> enc_ecd_x_copy(num_col);
    // s3_input_copy_fused: the stock code copies the whole 768-wide array at full level and only
    // then shrinks it, so both arrays are momentarily fresh at 768 x 36 MiB each and the pool
    // reserves 53,792 MiB to end up holding 27,648. Shrinking element i before copying element
    // i+1 demands the same values in the same per-element order and never needs the second array
    // at full width. `enc_ecd_x` is not switched until the loops below, so each copy is still
    // taken at exactly the level it had before.
    for (int i = 0; i < num_col; ++i){
        enc_ecd_x_copy[i] = enc_ecd_x[i];
        for (int j = 0; j < boot_level; ++j){
            evaluator.mod_switch_to_next_inplace(enc_ecd_x_copy[i]);
        }
    }
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "s3_input_copy_fused" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the copy-then-switch pair matched {txt.count(ANCHOR)} times, expected 1")

    at = txt.index(ANCHOR)

    # The source must not be switched above the copy: the equivalence argument is that every copy is
    # taken at the level enc_ecd_x had before any of its own switches.
    head = txt[:at]
    if "mod_switch_to_next_inplace(enc_ecd_x[" in head:
        sys.exit("FAILED: `enc_ecd_x` is already mod-switched above the copy; the level arithmetic "
                 "this step is derived from does not describe this tree")

    # boot_level must be defined above (a loop bound, not something the body changes).
    if "boot_level" not in head:
        sys.exit("FAILED: `boot_level` is not defined above this point")

    txt = txt.replace(ANCHOR, PATCH, 1)
    open(src, "w").write(txt)
    print("patch_input_copy_fused: the input copy is shrunk per element (" + SRC + ")")


if __name__ == "__main__":
    main()
