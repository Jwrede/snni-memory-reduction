#!/usr/bin/env python3
"""MOAI-CPU x_copy_release_early: free the residual copy right after the first residual add instead of
after layernorm1.

    patch_x_copy_release_early.py <moai source tree>

enc_ecd_x_copy (768 cts, 16.9 GB) is last read by `rtn[i] += enc_ecd_x_copy[i]` after bootstrap1;
x_copy_release (m3) frees it after layernorm1. m22's LN1 window (63.0 GiB, m22j table): three 16.9 GB
rows (rtn, LN1 working array, the dead copy). Release moved to the line after the add. Same bytes, free
(the later release block finds an empty vector, reports released=0).
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

ANCHOR = """    //rtn+enc_ecd_x_copy
    snni_unpark_cts(enc_ecd_x_copy, context, "residual1");
    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i){
        evaluator.mod_switch_to_inplace(enc_ecd_x_copy[i], rtn[i].parms_id());
        evaluator.add_inplace(rtn[i],enc_ecd_x_copy[i]);
    }
"""
ADDED = ANCHOR + """    // x_copy_release_early: the add above was the copy's last read; free it before layernorm1
    // runs instead of after it (the block after the layernorm then releases an empty vector).
    {
        size_t _snni_n = enc_ecd_x_copy.size();
        vector<Ciphertext>().swap(enc_ecd_x_copy);
        malloc_trim(0);
        fprintf(stderr, "LEVER|x_copy_release_early|released=%zu\\n", _snni_n);
        fflush(stderr);
    }
"""
LATER = 'fprintf(stderr, "LEVER|x_copy_release|released=%zu\\n", _snni_released);'


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "x_copy_release_early" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the residual-add anchor appears {txt.count(ANCHOR)} times, expected 1")
    if txt.count(LATER) != 1:
        sys.exit("FAILED: the later x_copy_release block is missing; this patch expects m3's release to exist")
    between = txt[txt.index(ANCHOR) + len(ANCHOR):txt.index(LATER)]
    reads = [l for l in between.split("\n") if "enc_ecd_x_copy" in l and "_snni_released = enc_ecd_x_copy.size()" not in l and "swap(enc_ecd_x_copy)" not in l]
    if reads:
        sys.exit("FAILED: enc_ecd_x_copy is still read between the residual add and the old release:\n" + "\n".join(reads))
    txt = txt.replace(ANCHOR, ADDED, 1)
    open(src, "w").write(txt)
    print("patch_x_copy_release_early: residual copy released right after the first residual add (" + SRC + ")")


if __name__ == "__main__":
    main()
