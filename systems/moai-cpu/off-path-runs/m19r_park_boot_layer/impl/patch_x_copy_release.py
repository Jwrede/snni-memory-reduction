#!/usr/bin/env python3
"""MOAI-CPU m2_x_copy_release: release enc_ecd_x_copy (768 residual-copy CTs) across the peak and re-create it before its next-layer reuse; two edits, both required. FREE.

    patch_x_copy_release.py <moai source tree>
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

RELEASE_ANCHOR = "    //vector<Ciphertext>().swap(enc_ecd_x_copy);\n"
RELEASE_PATCH = """    // LEVER x_copy_release: the residual copy is dead from its last read until it is reused in
    // the next layer, and the peak falls inside that interval. Shipped commented out because the
    // reuse below needs the vector to exist; the resize before that loop is the other half.
    {
        size_t _snni_released = enc_ecd_x_copy.size();
        vector<Ciphertext>().swap(enc_ecd_x_copy);
        fprintf(stderr, "LEVER|x_copy_release|released=%zu\\n", _snni_released);
        fflush(stderr);
    }
"""

REUSE_ANCHOR = """            enc_ecd_x[i*6+j] = rtn2[i*6+j];
            enc_ecd_x_copy[i*6+j] = rtn2[i*6+j];"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "x_copy_release" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(RELEASE_ANCHOR) != 1:
        sys.exit(f"FAILED: the commented release matched {txt.count(RELEASE_ANCHOR)} times, "
                 "expected exactly 1")
    if txt.count(REUSE_ANCHOR) != 1:
        sys.exit(f"FAILED: the reuse loop matched {txt.count(REUSE_ANCHOR)} times, expected 1. "
                 "The release must not be applied without the resize that precedes this loop.")

    # The release must precede the reuse (checked on offsets), or the resize is pointless and the
    # release becomes a use-after-free.
    if txt.index(RELEASE_ANCHOR) > txt.index(REUSE_ANCHOR):
        sys.exit("FAILED: the commented release appears AFTER the reuse loop in this tree; the "
                 "lifetime this step rests on does not hold here")

    # The last read (residual addition) must precede the release point.
    read = "evaluator.add_inplace(rtn[i],enc_ecd_x_copy[i]);"
    if txt.count(read) != 1:
        sys.exit(f"FAILED: the residual addition matched {txt.count(read)} times, expected 1")
    if txt.index(read) > txt.index(RELEASE_ANCHOR):
        sys.exit("FAILED: the residual addition is AFTER the release point in this tree")

    txt = txt.replace(RELEASE_ANCHOR, RELEASE_PATCH, 1)
    txt = txt.replace(REUSE_ANCHOR,
                      "            enc_ecd_x[i*6+j] = rtn2[i*6+j];\n"
                      "            enc_ecd_x_copy[i*6+j] = rtn2[i*6+j];", 1)

    # Resize immediately before the reuse loop's `#pragma omp parallel for`, not inside it: growing
    # a vector from several threads at once is a data race.
    head, _, tail = txt.partition(REUSE_ANCHOR)
    pragma = head.rfind("#pragma omp parallel for")
    if pragma < 0:
        sys.exit("FAILED: no `#pragma omp parallel for` before the reuse loop")
    txt = (head[:pragma]
           + "// LEVER x_copy_release: re-create what the release above gave back, before the\n"
           + "    // parallel region rather than inside it.\n"
           + "    enc_ecd_x_copy.resize(num_col);\n\n    "
           + head[pragma:] + REUSE_ANCHOR + tail)

    open(src, "w").write(txt)
    out = open(src).read()
    for what in ("LEVER|x_copy_release", "enc_ecd_x_copy.resize(num_col);"):
        if what not in out:
            sys.exit(f"FAILED: applied the patch but {what!r} is absent")
    if out.index("vector<Ciphertext>().swap(enc_ecd_x_copy);") > out.index(
            "enc_ecd_x_copy.resize(num_col);"):
        sys.exit("FAILED: the release ended up after the resize")
    print("patch_x_copy_release: residual copy released across the peak and re-created "
          "before its reuse (" + SRC + ")")


if __name__ == "__main__":
    main()
