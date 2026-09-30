#!/usr/bin/env python3
"""MOAI-CPU m12_enc_X_v_shrink: exactly-sized block for the V-input copy in single_att_block.

    patch_enc_X_v_shrink.py <moai source tree>

enc_X copied into enc_X_v (768 cts, 15 limbs, 15.7 MB each), switched in place to chain index 3
(4 limbs, 4.2 MB); capacity kept (dynarray.h:457): 12.1 GB resident per head, 3.2 GB used. Corrected m8
table: enc_X_v 12.1 GB, rank 6; after m9/m11 the largest attention-phase object with a free fix.
Technique of m9 (copy-assign by size, move back). Bit-identical.
"""
import os
import sys

SRC = "include/source/att_block/single_att_block.hpp"

INCLUDE_ANCHOR = "#include <chrono>\n"
INCLUDE = "#include <chrono>\n#include <malloc.h>\n"

HELPER_ANCHOR = "vector<Ciphertext> single_att_block("
HELPER = """// m12_enc_X_v_shrink: SEAL keeps a DynArray's capacity when it is resized smaller; copy-assign
// allocates by size, so a copy into a fresh Ciphertext and a move back gives an exactly-sized block.
static void snni_shrink_cts_att(vector<Ciphertext> &v) {
    size_t cap_before = v.empty() ? 0 : v[0].dyn_array().capacity();
    #pragma omp parallel for
    for (size_t i = 0; i < v.size(); ++i) {
        Ciphertext t;
        t = v[i];
        v[i] = std::move(t);
    }
    malloc_trim(0);
    static bool _snni_once = false;
    if (!_snni_once) {
        _snni_once = true;
        fprintf(stderr, "LEVER|enc_X_v_shrink|n=%zu|limbs=%zu|capacity_u64_before=%zu|after=%zu\\n",
                v.size(), v.empty() ? (size_t)0 : v[0].coeff_modulus_size(), cap_before,
                v.empty() ? (size_t)0 : v[0].dyn_array().capacity());
        fflush(stderr);
    }
}

vector<Ciphertext> single_att_block("""

LOOP_ANCHOR = """      while (seal_context.get_context_data(enc_X_v[i].parms_id())->chain_index()>3){
        evaluator.mod_switch_to_next_inplace(enc_X_v[i]);
      }
  }
"""
LOOP = LOOP_ANCHOR + "  snni_shrink_cts_att(enc_X_v);\n"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "enc_X_v_shrink" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, anchor in (("include", INCLUDE_ANCHOR), ("helper", HELPER_ANCHOR), ("loop", LOOP_ANCHOR)):
        n = txt.count(anchor)
        if n != 1:
            sys.exit(f"FAILED: the {name} anchor appears {n} times, expected 1")
    txt = txt.replace(INCLUDE_ANCHOR, INCLUDE, 1)
    txt = txt.replace(HELPER_ANCHOR, HELPER, 1)
    txt = txt.replace(LOOP_ANCHOR, LOOP, 1)
    open(src, "w").write(txt)
    if open(src).read().count("snni_shrink_cts_att(") != 2:
        sys.exit("FAILED: applied but the helper is not called once")
    print("patch_enc_X_v_shrink: enc_X_v shrunk to its level (" + SRC + ")")


if __name__ == "__main__":
    main()
