#!/usr/bin/env python3
"""MOAI-GPU s5: LN1 residual copy parked off the device for the whole attention block. Paid.

    patch_park_input_copy.py <moai source tree>

First paid step (free levers exhausted). before_attention (49.6% of peak) floor = two live arrays;
enc_ecd_x_copy (768 x 21 MiB) unread until the add loop after bootstrap1: parked to the host disk
bank, reloaded one ciphertext at a time. Paid. Expected up to -16,128 MiB (14.9% of 108,384 MiB).
Mechanism from retired w1_park_residuals (its boot_layer half is s3_boot_layer_release, free).
MOAI_SPILL_DIR-gated. Gate T1s (HostCipher carries the full ciphertext state).
"""
import os
import re
import sys

INC = "src/include/include.cuh"
SRC = "src/include/test/test_single_layer.cuh"
HELPERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "borrowed", "wpark_helpers.cuh")

INC_ANCHOR = "// source code\n"

PARK_ANCHOR = """    cout << "Modulus chain index before attention block: " << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth() << endl;
    snni_mem_marker("before_attention");
"""

PARK = """    cout << "Modulus chain index before attention block: " << context.get_context_data(enc_ecd_x[0].params_id()).chain_depth() << endl;
    // s5_park_input_copy: the LN1 residual is not read again until the add loop after bootstrap1,
    // and it is 768 x 21 MiB of device memory sitting through the entire attention block. Park it
    // off the device here; with MOAI_SPILL_DIR set the payload goes straight to a disk bank and
    // only metadata stays in host RAM, so neither pool pays for it.
    auto enc_ecd_x_copy_host = offload_vector_to_host(enc_ecd_x_copy);
    std::cerr << "LEVER|park_input_copy|parked_ciphertexts=" << enc_ecd_x_copy_host.size()
              << "|disk=" << (std::getenv("MOAI_SPILL_DIR") ? 1 : 0) << std::endl;
    snni_mem_marker("before_attention");
"""

RELOAD_ANCHOR = """    for (int i = 0; i < num_col; ++i){
        evaluator.mod_switch_to_inplace(enc_ecd_x_copy[i], rtn[i].params_id());
        evaluator.add_inplace(rtn[i],enc_ecd_x_copy[i]);
    }
"""

RELOAD = """    // s5_park_input_copy: reload ONE ciphertext at a time, so the residual is never fully device
    // resident again. Same arithmetic in the same order; only the source of each operand changed.
    for (int i = 0; i < num_col; ++i){
        PhantomCiphertext _res;
        reload_cipher_from_host(_res, enc_ecd_x_copy_host[i]);
        evaluator.mod_switch_to_inplace(_res, rtn[i].params_id());
        evaluator.add_inplace(rtn[i], _res);
    }
    std::vector<HostCipher>().swap(enc_ecd_x_copy_host);
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tree = sys.argv[1]
    inc = os.path.join(tree, INC)
    src = os.path.join(tree, SRC)
    for p in (inc, src):
        if not os.path.exists(p):
            sys.exit("FAILED: no " + p)
    if not os.path.exists(HELPERS):
        sys.exit("FAILED: no borrowed/wpark_helpers.cuh beside this patch; the mechanism is taken "
                 "verbatim from the retired image and is not reconstructed here")

    itxt = open(inc).read()
    stxt = open(src).read()

    # The lever and the infrastructure are checked separately: the helpers are shared infrastructure
    # (a tree may already carry them via patch_park_helpers.py), only the lever may exist once.
    if "s5_park_input_copy" in stxt:
        sys.exit("FAILED: this tree already carries the lever")
    _have_helpers = "offload_vector_to_host" in itxt

    # The helpers go before the source includes (Ct_pt_matrix_mul.cuh etc. are compiled against them).
    if _have_helpers:
        print("note: the helpers are already in include.cuh, inserting the lever only")
    elif itxt.count(INC_ANCHOR) != 1:
        sys.exit(f"FAILED: the `// source code` marker matched {itxt.count(INC_ANCHOR)} times in "
                 "include.cuh, expected exactly 1")

    helpers = "" if _have_helpers else open(HELPERS).read()
    # The borrowed block must contain the two entry points this step calls and the env gate.
    for need in ("offload_vector_to_host", "reload_cipher_from_host", "MOAI_SPILL_DIR"):
        if need not in (itxt if _have_helpers else helpers):
            sys.exit(f"FAILED: the park helpers do not contain `{need}` "
                     f"({'already in include.cuh' if _have_helpers else 'borrowed/wpark_helpers.cuh'})")

    if "#include <fcntl.h>" not in itxt:
        if "#include <unistd.h>" not in itxt:
            sys.exit("FAILED: include.cuh has neither fcntl.h nor unistd.h; the bank helpers need "
                     "open/pread/pwrite and this tree is not the one they were taken from")
        itxt = itxt.replace("#include <unistd.h>", "#include <unistd.h>\n#include <fcntl.h>", 1)

    if not _have_helpers:
        itxt = itxt.replace(INC_ANCHOR,
                            "// ---- borrowed verbatim from the retired w1_park_residuals image "
                            "(include.cuh:227-454) ----\n" + helpers +
                            "// ---- end borrowed block ----\n\n" + INC_ANCHOR, 1)

    # Park site: marker kept AFTER the park so before_attention measures the parked state.
    if stxt.count(PARK_ANCHOR) != 1:
        sys.exit(f"FAILED: the park site matched {stxt.count(PARK_ANCHOR)} times, expected 1")
    # Reload site must be the LN1 add loop, not the LN2 one (boot_layer's loop looks almost identical
    # and is already handled by s3_boot_layer_release).
    if stxt.count(RELOAD_ANCHOR) != 1:
        sys.exit(f"FAILED: the LN1 add loop matched {stxt.count(RELOAD_ANCHOR)} times, expected 1")

    # Nothing may read enc_ecd_x_copy between the park and the reload (the whole claim).
    at_park = stxt.index(PARK_ANCHOR) + len(PARK_ANCHOR)
    at_reload = stxt.index(RELOAD_ANCHOR)
    if at_reload < at_park:
        sys.exit("FAILED: the reload site sits BEFORE the park site")
    # Comments stripped first: otherwise the stock comment //rtn+enc_ecd_x_copy above the add loop
    # (a mention, not a read) trips this check.
    between = re.sub(r"//[^\n]*", "", stxt[at_park:at_reload])
    if "enc_ecd_x_copy" in between:
        sys.exit("FAILED: `enc_ecd_x_copy` is read between the park and the reload; parking it "
                 "there would lose something still in use")

    stxt = stxt.replace(PARK_ANCHOR, PARK, 1)
    stxt = stxt.replace(RELOAD_ANCHOR, RELOAD, 1)

    open(inc, "w").write(itxt)
    open(src, "w").write(stxt)
    print("patch_park_input_copy: LN1 residual parked off the device for the attention block")
    print("  helpers inserted into " + INC + " (" + str(len(helpers.splitlines())) + " borrowed lines)")


if __name__ == "__main__":
    main()
