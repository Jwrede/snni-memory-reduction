#!/usr/bin/env python3
"""MOAI-CPU m6_att_output_release: release att_output after its only consumer (the att_selfoutput matmul); dead from there and never reused, so one edit. FREE. Base = m5_boot_layer_release.

    patch_att_output_release.py <moai source tree>
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

# att_output's only read is the ct_pt_matrix_mul that builds att_selfoutput. It is never reused, and
# the peak (after_bootstrap4) is far later. One edit: swap it right after that matmul returns.
ANCHOR = (
    "    vector<Ciphertext> att_selfoutput = ct_pt_matrix_mul_wo_pre_w_mask(att_output, selfoutput, b_vec, num_col, num_col, num_col, context);\n"
    "    int att_selfoutput_size = att_selfoutput.size();\n"
)
PATCH = (
    "    vector<Ciphertext> att_selfoutput = ct_pt_matrix_mul_wo_pre_w_mask(att_output, selfoutput, b_vec, num_col, num_col, num_col, context);\n"
    "    // LEVER att_output_release: att_output is dead after this matmul (its only consumer) and never\n"
    "    // reused; the peak (after_bootstrap4) is far later. No resize needed (like boot_layer).\n"
    "    {\n"
    "        size_t _snni_released = att_output.size();\n"
    "        vector<Ciphertext>().swap(att_output);\n"
    '        fprintf(stderr, "LEVER|att_output_release|released=%zu\\n", _snni_released);\n'
    "        fflush(stderr);\n"
    "    }\n"
    "    int att_selfoutput_size = att_selfoutput.size();\n"
)

LAST_READ = "ct_pt_matrix_mul_wo_pre_w_mask(att_output,"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "att_output_release" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if "vector<Ciphertext>().swap(att_output);" in txt:
        sys.exit("FAILED: this tree already swaps att_output")
    if txt.count(LAST_READ) != 1:
        sys.exit(f"FAILED: att_output's matmul read matched {txt.count(LAST_READ)} times, expected 1")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the att_selfoutput anchor matched {txt.count(ANCHOR)} times, expected 1")
    # att_output must not be used after the release point.
    after = txt[txt.index(ANCHOR) + len(ANCHOR):]
    if "att_output" in after:
        sys.exit("FAILED: att_output is used after the release point; a swap here is a use-after-free")

    txt = txt.replace(ANCHOR, PATCH, 1)
    open(src, "w").write(txt)
    out = open(src).read()
    if "LEVER|att_output_release" not in out or "swap(att_output);" not in out:
        sys.exit("FAILED: applied the patch but the marker or swap is absent")
    print("patch_att_output_release: att_output released after its last use (" + SRC + ")")


if __name__ == "__main__":
    main()
