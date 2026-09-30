#!/usr/bin/env python3
"""MOAI-CPU m1_minimal_att_keys: gal_keys is read only by Ct_ct_matrix_mul at multiples of num_X, whose NAF has no term below num_X, so 17 of 31 attention Galois keys (~24.1 GB) are unreachable. FREE.

    patch_minimal_att_keys.py <moai source tree>
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

ANCHOR = "    keygen.create_galois_keys(gal_keys);"

PATCH = """    // m1_minimal_att_keys: `gal_keys` is read ONLY by Ct_ct_matrix_mul, and only at multiples
    // of num_batch (= num_X). SEAL's NAF decomposition of 256k has terms +-2^(8+m) only, so no
    // step below num_X can ever be requested and no conjugation is needed on this path. The
    // default set generates 31 elements; 14 are reachable. See impl/patch_minimal_att_keys.py.
    vector<int> att_steps;
    const int snni_max_rot = 1 << (logN - 2);
    for (int snni_step = num_X; snni_step <= snni_max_rot; snni_step <<= 1) {
        att_steps.push_back(snni_step);
        att_steps.push_back(-snni_step);
    }
    keygen.create_galois_keys(att_steps, gal_keys);
    // `logN` is a `long` here, so the default-set size is cast rather than handed to %d as-is.
    // The same class of slip cost a BumbleBee image build an hour earlier today, where
    // -Werror=format caught it; this build has no -Werror and would have printed garbage instead.
    fprintf(stderr, "LEVER|minimal_att_keys|requested=%zu|default_would_be=%d|num_X=%d|max_rot=%d\\n",
            att_steps.size(), (int)(1 + 2 * (logN - 1)), num_X, snni_max_rot);
    fflush(stderr);
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "minimal_att_keys" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the default key generation matched {txt.count(ANCHOR)} times, expected 1")

    # The bound depends on num_X and logN being in scope and being what the proof assumes; a
    # renamed num_X or a different num_batch would make the bound wrong while still compiling.
    if "const int num_X = " not in txt:
        sys.exit("FAILED: no `num_X` declaration; the lower bound of the reachable set is derived "
                 "from it")
    if "gal_keys, bootstrapper, num_X" not in txt.replace("\n", " ").replace("  ", " "):
        sys.exit("FAILED: the attention is not handed `num_X` as its num_batch in this tree, so "
                 "the proof that every rotation is a multiple of num_X does not hold here. "
                 "Re-derive the bound rather than loosening it.")

    txt = txt.replace(ANCHOR, PATCH, 1)
    open(src, "w").write(txt)
    for what in ("att_steps", "LEVER|minimal_att_keys"):
        if what not in open(src).read():
            sys.exit(f"FAILED: applied the patch but {what!r} is absent")
    print("patch_minimal_att_keys: attention Galois keys reduced to the reachable set (" + SRC + ")")


if __name__ == "__main__":
    main()
