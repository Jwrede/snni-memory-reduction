#!/usr/bin/env python3
"""MOAI-GPU s6: LN1 residual never exists as a device array. Paid.

    patch_fused_park_input.py <moai source tree>

From two nulls: s4_input_copy_fused (before_attention 53,792 -> 43,136, run total unchanged) and
s5_park_input_copy (-16,128 MiB used, unchanged). Copy one element, mod-switch it down, park it, then
the next (36 MiB on device at a time, 0 resident). Expected up to -26,000 MiB (~-24% of 108,384 MiB).
Paid (disk bank, read back in the LN1 add loop; s5's reload side applied first). Gate T1s.
"""
import os
import sys

SRC = "src/include/test/test_single_layer.cuh"

# The stock producing pair, unchanged from the baseline: build the whole array, then shrink it.
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

PATCH = """    // s6_fused_park_input: the residual never exists as a device array. Each element is copied,
    // shrunk and parked before the next one is touched, so the device holds ONE ciphertext of it
    // at a time instead of 768. Same values, same per-element order, same mod-switch count; only
    // the lifetime of the array changed. `enc_ecd_x` is untouched until the loops below, so every
    // copy is still taken at exactly the level it had before.
    std::vector<HostCipher> enc_ecd_x_copy_host;
    enc_ecd_x_copy_host.reserve(num_col);
    for (int i = 0; i < num_col; ++i){
        PhantomCiphertext _one = enc_ecd_x[i];
        for (int j = 0; j < boot_level; ++j){
            evaluator.mod_switch_to_next_inplace(_one);
        }
        enc_ecd_x_copy_host.push_back(offload_cipher_to_host(_one));
    }
    cudaDeviceSynchronize();
    std::cerr << "LEVER|fused_park_input|parked_ciphertexts=" << enc_ecd_x_copy_host.size()
              << "|disk=" << (std::getenv("MOAI_SPILL_DIR") ? 1 : 0) << std::endl;
"""

# s5's park line, which this step replaces: the array it parked no longer exists.
S5_PARK = """    auto enc_ecd_x_copy_host = offload_vector_to_host(enc_ecd_x_copy);
    std::cerr << "LEVER|park_input_copy|parked_ciphertexts=" << enc_ecd_x_copy_host.size()
              << "|disk=" << (std::getenv("MOAI_SPILL_DIR") ? 1 : 0) << std::endl;
"""

# The stock line that frees the array after the LN1 add loop; removed since no array is left.
STOCK_SWAP ="""    vector<PhantomCiphertext>().swap(enc_ecd_x_copy);
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "s6_fused_park_input" in txt:
        sys.exit("FAILED: this tree already carries the lever")

    # s5 must be applied first: this step edits only the producing loop and relies on s5's reload
    # side, else the residual would be parked and never read back.
    if "reload_cipher_from_host(_res, enc_ecd_x_copy_host[i])" not in txt:
        sys.exit("FAILED: this tree does not carry s5_park_input_copy's reload loop; this step "
                 "edits only the producing side and would park a residual nobody reads back")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the stock copy-then-shrink pair matched {txt.count(ANCHOR)} times, "
                 "expected exactly 1")
    if txt.count(S5_PARK) != 1:
        sys.exit(f"FAILED: s5's park line matched {txt.count(S5_PARK)} times, expected exactly 1")
    if txt.count(STOCK_SWAP) != 1:
        sys.exit(f"FAILED: the stock swap matched {txt.count(STOCK_SWAP)} times, expected exactly 1")

    # ORDER MATTERS: the producing loop comes first, so patch it first and then remove s5's park.
    if txt.index(ANCHOR) > txt.index(S5_PARK):
        sys.exit("FAILED: s5's park sits BEFORE the producing loop; this tree is not shaped the "
                 "way this step assumes")

    txt = txt.replace(ANCHOR, PATCH, 1)
    # s5 parked the finished array; nothing is left to park, and leaving the call would redeclare it.
    txt = txt.replace(S5_PARK, "", 1)
    txt = txt.replace(STOCK_SWAP, "", 1)

    # The name must be gone entirely, not just its subscripted uses: an earlier guard looked for
    # `enc_ecd_x_copy[` and passed a tree still carrying the subscript-less swap, which failed to build.
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("//") or "enc_ecd_x_copy" not in line:
            continue
        if "enc_ecd_x_copy_host" in line and "enc_ecd_x_copy[" not in line:
            # the host-side vector is this step's own and is allowed
            rest = line.replace("enc_ecd_x_copy_host", "")
            if "enc_ecd_x_copy" not in rest:
                continue
        sys.exit("FAILED: `enc_ecd_x_copy` is still referenced, and this step no longer declares "
                 "it:\n  " + s[:110])

    open(src, "w").write(txt)
    print("patch_fused_park_input: the LN1 residual is copied, shrunk and parked one element at a time")


if __name__ == "__main__":
    main()
