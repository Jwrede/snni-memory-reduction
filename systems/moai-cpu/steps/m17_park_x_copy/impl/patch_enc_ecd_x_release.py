#!/usr/bin/env python3
"""MOAI-CPU m4_enc_ecd_x_release: release enc_ecd_x (768 input CTs) across the peak and re-create it before its next-layer reuse; two edits, both required. FREE. Base = m3_x_copy_release.

    patch_enc_ecd_x_release.py <moai source tree>
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

# enc_ecd_x's last read is the attention block; it is dead from there until its layer-2 reuse, and the
# peak (after_bootstrap4) lies inside that interval. Released right after the after_attention marker.
RELEASE_ANCHOR = '    snni_mark("after_attention");\n'
RELEASE_PATCH = '''    snni_mark("after_attention");
    // LEVER enc_ecd_x_release: the input is dead from its last read (single_att_block above) until it
    // is reused in the next layer, and the peak falls inside that interval. The resize before the
    // reuse loop is the other half.
    {
        size_t _snni_released = enc_ecd_x.size();
        vector<Ciphertext>().swap(enc_ecd_x);
        fprintf(stderr, "LEVER|enc_ecd_x_release|released=%zu\\n", _snni_released);
        fflush(stderr);
    }
'''

# The reuse loop re-creates enc_ecd_x AND enc_ecd_x_copy; m3 already resized the copy here. Ties this
# step to an m3 base (fails on a pristine/m2 tree, which lacks this line).
RESIZE_ANCHOR = "    enc_ecd_x_copy.resize(num_col);\n"
RESIZE_PATCH = "    enc_ecd_x.resize(num_col);\n    enc_ecd_x_copy.resize(num_col);\n"

LAST_READ = "single_att_block(enc_ecd_x,"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "enc_ecd_x_release" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if "vector<Ciphertext>().swap(enc_ecd_x);" in txt:
        sys.exit("FAILED: this tree already swaps enc_ecd_x")
    for name, s, n in (("last read", LAST_READ, txt.count(LAST_READ)),
                       ("after_attention release anchor", RELEASE_ANCHOR, txt.count(RELEASE_ANCHOR)),
                       ("m3 copy-resize anchor", RESIZE_ANCHOR, txt.count(RESIZE_ANCHOR))):
        if n != 1:
            sys.exit(f"FAILED: {name} matched {n} times, expected 1 "
                     "(is the base image m3_x_copy_release?)")

    # last read must precede the release point, release must precede the reuse/resize.
    if txt.index(LAST_READ) > txt.index(RELEASE_ANCHOR):
        sys.exit("FAILED: enc_ecd_x's last read is AFTER the release point in this tree")
    if txt.index(RELEASE_ANCHOR) > txt.index(RESIZE_ANCHOR):
        sys.exit("FAILED: the release point is AFTER the reuse loop in this tree")

    txt = txt.replace(RELEASE_ANCHOR, RELEASE_PATCH, 1)
    txt = txt.replace(RESIZE_ANCHOR, RESIZE_PATCH, 1)

    open(src, "w").write(txt)
    out = open(src).read()
    for what in ("LEVER|enc_ecd_x_release", "enc_ecd_x.resize(num_col);"):
        if what not in out:
            sys.exit(f"FAILED: applied the patch but {what!r} is absent")
    if out.index("vector<Ciphertext>().swap(enc_ecd_x);") > out.index("enc_ecd_x.resize(num_col);"):
        sys.exit("FAILED: the release ended up after the resize")
    print("patch_enc_ecd_x_release: input released across the peak and re-created before its reuse ("
          + SRC + ")")


if __name__ == "__main__":
    main()
