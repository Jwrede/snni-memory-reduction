#!/usr/bin/env python3
"""MOAI-CPU m15_copy_after_switch: take the residual copy after the input is switched to the
bootstrap level (pre-attention transient no longer holds two full-chain arrays).

    patch_copy_after_switch.py <moai source tree>

Basis: m14 line peaks at run_begin, 75.3 GiB: keys 21 + enc_ecd_x 28 + enc_ecd_x_copy 28, both at the
full 35-limb chain. Switch enc_ecd_x down, shrink (m9's helper), then copy: copy at 22 MB per
ciphertext; transient ~55 GiB. Bit-identical (mod_switch_to_next deterministic; copy-then-switch =
switch-then-copy). Free (768 fewer mod switches). Requires the m14 tree (anchors: m9 shrink, m14 park).
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

# 1. The copy loop right after batch_input: keep the declaration, drop the copy.
CREATE_ANCHOR = """    vector<Ciphertext> enc_ecd_x_copy(num_col);
    for (int i = 0; i < num_col; ++i){
        enc_ecd_x_copy[i] = enc_ecd_x[i];
    }
"""
CREATE = "    vector<Ciphertext> enc_ecd_x_copy(num_col);\n"

# 2. The copy's own switch loop with m9's shrink and m14's park: removed here, re-created below.
COPYLOOP_ANCHOR = """    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i) {
        for (int j = 0; j < boot_level; ++j){
            evaluator.mod_switch_to_next_inplace(enc_ecd_x_copy[i]);
        }
    }
    snni_shrink_cts(enc_ecd_x_copy, "enc_ecd_x_copy");
    snni_park_cts(enc_ecd_x_copy, "layer0");
"""
COPYLOOP = ""

# 3. After the input's switch loop to the bootstrap level: shrink, copy, park.
XLOOP_ANCHOR = """        for (int j = 0; j < boot_level; ++j){
            evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
        }
    }
"""
XLOOP = XLOOP_ANCHOR + """    // m15_copy_after_switch: the input is at the bootstrap level now; shrink it, then copy it
    // (copy-assign allocates by size, so the copy needs no shrink of its own), then park the copy.
    snni_shrink_cts(enc_ecd_x, "enc_ecd_x_boot_level");
    for (int i = 0; i < num_col; ++i){
        enc_ecd_x_copy[i] = enc_ecd_x[i];
    }
    fprintf(stderr, "LEVER|copy_after_switch|copy_limbs=%zu|copy_capacity_u64=%zu\\n",
            enc_ecd_x_copy[0].coeff_modulus_size(), enc_ecd_x_copy[0].dyn_array().capacity());
    fflush(stderr);
    snni_park_cts(enc_ecd_x_copy, "layer0");
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "copy_after_switch" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, anchor in (("create", CREATE_ANCHOR), ("copy loop", COPYLOOP_ANCHOR), ("x loop", XLOOP_ANCHOR)):
        n = txt.count(anchor)
        if n != 1:
            sys.exit(f"FAILED: the {name} anchor appears {n} times, expected 1")
    txt = txt.replace(CREATE_ANCHOR, CREATE, 1)
    txt = txt.replace(COPYLOOP_ANCHOR, COPYLOOP, 1)
    txt = txt.replace(XLOOP_ANCHOR, XLOOP, 1)
    open(src, "w").write(txt)
    out = open(src).read()
    if out.count('snni_park_cts(enc_ecd_x_copy, "layer0")') != 1 or out.count("enc_ecd_x_copy[i] = enc_ecd_x[i];") != 1:
        sys.exit("FAILED: applied but the copy/park are not exactly once")
    print("patch_copy_after_switch: residual copy taken at the bootstrap level (" + SRC + ")")


if __name__ == "__main__":
    main()
