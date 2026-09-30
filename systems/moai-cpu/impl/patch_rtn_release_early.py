#!/usr/bin/env python3
"""MOAI-CPU m2_rtn_release_early: release `rtn` (768 CTs) at its last read (the inter matmul), not 238 lines later; reaches the OS only because m1_pool_new dropped SEAL's retaining pool. FREE.

    patch_rtn_release_early.py <path to the moai source tree>
"""
import os
import re
import sys

ANCHOR = ("    vector<Ciphertext> inter_output = ct_pt_matrix_mul_wo_pre_large("
          "rtn, inter_weight, num_col, num_inter, num_col, context);")

ADDED = """
    // m2_rtn_release_early: `rtn` is dead from here. Its last reader is the call above, which takes
    // it by const reference and has returned. The stock code keeps all 768 ciphertexts alive until
    // the swap at the end of the layer, across the FFN, the third bootstrap and the second
    // layernorm -- which is where this run's peak falls (99.0% of the run, from the poller trace).
    //
    // This only reaches the OS because m1_pool_new replaced SEAL's global retaining pool with
    // MMProfNew. Under the stock pool the same release would return pages to the pool and the
    // peak, which is the pool's high-water, would not move.
    vector<Ciphertext>().swap(rtn);
    std::cerr << "LEVER|rtn_release_early|freed_ciphertexts=" << 768 << std::endl;
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], "include/test/test_full_scheme.hpp")
    if not os.path.exists(src):
        sys.exit("FAILED: no test_full_scheme.hpp under " + sys.argv[1])
    txt = open(src).read()

    if "LEVER|rtn_release_early" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit("FAILED: the last read of `rtn` is not present exactly once "
                 f"(found {txt.count(ANCHOR)}); this patch expects the m1 tree")

    # Refuse if `rtn` is read after the anchor (comments stripped first, since the two mentions
    # before the release are comments); releasing a live value would be wrong.
    tail = txt[txt.index(ANCHOR) + len(ANCHOR):]
    tail = re.sub(r"//[^\n]*", "", tail)
    tail = re.sub(r"/\*.*?\*/", "", tail, flags=re.S)
    tail_before_swap = tail.split("vector<Ciphertext>().swap(rtn);")[0]
    uses = re.findall(r"\brtn\b", tail_before_swap)
    if uses:
        sys.exit(f"FAILED: `rtn` is read {len(uses)} more time(s) after its supposed last use; "
                 "releasing it here would free something still live")

    txt = txt.replace(ANCHOR, ANCHOR + ADDED, 1)
    open(src, "w").write(txt)
    print("patch_rtn_release_early: rtn released at its last read (include/test/test_full_scheme.hpp)")


if __name__ == "__main__":
    main()
