#!/usr/bin/env python3
"""MOAI-CPU m9_ct_capacity_shrink: exactly-sized blocks for the two input ciphertext arrays.

    patch_ct_capacity_shrink.py <moai source tree>

SEAL DynArray::resize keeps capacity on shrink (dynarray.h:457, "if (size <= capacity_)").
enc_ecd_x: encrypted at 35 limbs (36.7 MB), switched in place to 15 limbs (15.7 MB); enc_ecd_x_copy:
copied at 35 limbs, switched to 21 limbs (22.0 MB). Both keep 36.7 MB blocks: 768 x (21.0 + 14.7) MB =
27.4 GB of slack resident in every phase (m8 live-pool dump: 36,700,160 B per item at chain index 14).
Copy-assignment allocates by size; copy into a fresh Ciphertext and move back = same bytes, exact block.
Bit-identical. Free (two parallel memcpy passes over 28 GB).
Where: (1) after the enc_ecd_x_copy mod-switch loop (before the layer loop); (2) before the
before_attention marker in the layer loop (layer 0's 35->15, later layers' 21->15). The layer-1
re-creation enc_ecd_x[k] = rtn2[k] already allocates by size.
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

HELPER_ANCHOR = "void all_layer_test(){"
HELPER = """// ---------------------------------------------------------------------------------------
// m9_ct_capacity_shrink: SEAL keeps a DynArray's capacity when it is resized smaller, so a
// ciphertext mod-switched in place keeps the block of the level it was created at. Copy-assign
// allocates by size; copying into a fresh Ciphertext and moving it back gives the same bytes in an
// exactly-sized block. Bit-identical.
static void snni_shrink_cts(std::vector<seal::Ciphertext> &v, const char *what) {
    size_t cap_before = v.empty() ? 0 : v[0].dyn_array().capacity();
    #pragma omp parallel for
    for (size_t i = 0; i < v.size(); ++i) {
        seal::Ciphertext t;
        t = v[i];
        v[i] = std::move(t);
    }
    malloc_trim(0);
    size_t cap_after = v.empty() ? 0 : v[0].dyn_array().capacity();
    fprintf(stderr, "LEVER|ct_capacity_shrink|%s|n=%zu|limbs=%zu|capacity_u64_before=%zu|after=%zu\\n",
            what, v.size(), v.empty() ? (size_t)0 : v[0].coeff_modulus_size(), cap_before, cap_after);
    fflush(stderr);
}
// ---------------------------------------------------------------------------------------

void all_layer_test(){"""

# (1) the copy's own mod-switch loop; unique through the array name inside it.
ANCHOR_COPY = """            evaluator.mod_switch_to_next_inplace(enc_ecd_x_copy[i]);
        }
    }
"""
SHRINK_COPY = ANCHOR_COPY + """    snni_shrink_cts(enc_ecd_x_copy, "enc_ecd_x_copy");
"""

# (2) the input at the attention level, every layer.
ANCHOR_ATT = '    snni_mark("before_attention");'
SHRINK_ATT = """    snni_shrink_cts(enc_ecd_x, "enc_ecd_x");
""" + ANCHOR_ATT


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "ct_capacity_shrink" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, anchor in (("helper", HELPER_ANCHOR), ("copy loop", ANCHOR_COPY), ("before_attention", ANCHOR_ATT)):
        n = txt.count(anchor)
        if n != 1:
            sys.exit(f"FAILED: the {name} anchor appears {n} times, expected 1")
    if "malloc_trim(0)" not in txt:
        sys.exit("FAILED: malloc_trim is not used in this tree yet (the phase-split port brings <malloc.h>); "
                 "add the include before applying")
    txt = txt.replace(HELPER_ANCHOR, HELPER, 1)
    txt = txt.replace(ANCHOR_COPY, SHRINK_COPY, 1)
    txt = txt.replace(ANCHOR_ATT, SHRINK_ATT, 1)
    open(src, "w").write(txt)
    out = open(src).read()
    if out.count("snni_shrink_cts(") != 3:
        sys.exit("FAILED: applied the patch but the helper is not called twice")
    print("patch_ct_capacity_shrink: enc_ecd_x_copy and enc_ecd_x shrunk to their level (" + SRC + ")")


if __name__ == "__main__":
    main()
