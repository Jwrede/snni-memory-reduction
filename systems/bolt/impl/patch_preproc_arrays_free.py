#!/usr/bin/env python3
"""BOLT s3: free the four per-row arrays the weight preprocessor allocates with raw new[] and never releases.
matrix_mod_p1/p2 are dead after the cross-packing call; matrix1/matrix2 are never touched. ~1019 MB over 12 layers, written half (510 MB) resident.
    patch_preproc_arrays_free.py <EzPC tree>
"""
import os
import re
import sys

SRC = "SCI/tests/bert_bolt/linear.cpp"

ANCHOR = """    pp.cross_mat_single = bert_cross_packing_single_matrix_2(he, matrix_mod_p1.data(), matrix_mod_p2.data(), data);
"""

PATCH = """    pp.cross_mat_single = bert_cross_packing_single_matrix_2(he, matrix_mod_p1.data(), matrix_mod_p2.data(), data);
    // s3_preproc_arrays_free: this function allocated four arrays per row with raw new[] into
    // vectors that own nothing, and freed none of them. matrix_mod_p1/p2 are dead the moment the
    // call above has read them; matrix1/matrix2 were never written or read at all. Over 12 layers
    // and 3 call sites that is 1019 MB that nothing ever released, of which the written half
    // (510 MB) is resident.
    for (int i = 0; i < common_dim; i++) {
        delete[] matrix_mod_p1[i];
        delete[] matrix_mod_p2[i];
        delete[] matrix1[i];
        delete[] matrix2[i];
    }
"""

MARKER_ANCHOR = """PreprocessParams_2 Linear::params_preprocessing_ct_pt(
"""

MARKER = """static bool _snni_preproc_free_announced = false;

PreprocessParams_2 Linear::params_preprocessing_ct_pt(
"""

ANNOUNCE_ANCHOR = """    PreprocessParams_2 pp;
    uint64_t plain_mod = he->plain_mod;
"""

ANNOUNCE = """    PreprocessParams_2 pp;
    uint64_t plain_mod = he->plain_mod;
    if (!_snni_preproc_free_announced) {
        _snni_preproc_free_announced = true;
        fprintf(stderr, "LEVER|preproc_arrays_free|armed\\n");
    }
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "s3_preproc_arrays_free" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, anchor in (("cross packing call", ANCHOR),
                         ("function signature", MARKER_ANCHOR),
                         ("function preamble", ANNOUNCE_ANCHOR)):
        if txt.count(anchor) != 1:
            sys.exit(f"FAILED: the {name} matched {txt.count(anchor)} times, expected exactly 1")

    # Verify the four arrays are allocated with new[] here and nothing already frees them; an
    # existing delete[] would mean the reading is stale and the patch a double free.
    body_start = txt.index(MARKER_ANCHOR)
    body_end = txt.index(ANCHOR) + len(ANCHOR)
    body = txt[body_start:body_end]
    for arr in ("matrix_mod_p1", "matrix_mod_p2", "matrix1", "matrix2"):
        if f"{arr}[i] = new uint64_t" not in body:
            sys.exit(f"FAILED: `{arr}` is not allocated with new[] here; the function is not the "
                     "one this step read")
    if "delete" in body:
        sys.exit("FAILED: this function already frees something. The step's entire claim is that "
                 "it frees nothing, so the reading is stale and the patch could double free")

    # Nothing may read the arrays after the cross-packing call (their asserted last reader);
    # comments are stripped first so the patch's own comment cannot satisfy the check.
    tail = txt[body_end:]
    nxt = re.search(r"\n\}", tail)
    tail_fn = tail[:nxt.start()] if nxt else tail
    tail_fn = re.sub(r"//[^\n]*", "", tail_fn)
    for arr in ("matrix_mod_p1", "matrix_mod_p2", "matrix1", "matrix2"):
        if arr in tail_fn:
            sys.exit(f"FAILED: `{arr}` is used after the cross packing call; it is not dead there "
                     "and freeing it would lose data still in use")

    txt = txt.replace(MARKER_ANCHOR, MARKER, 1)
    txt = txt.replace(ANNOUNCE_ANCHOR, ANNOUNCE, 1)
    txt = txt.replace(ANCHOR, PATCH, 1)

    if "#include <cstdio>" not in txt and "#include <stdio.h>" not in txt:
        txt = "#include <cstdio>\n" + txt

    open(src, "w").write(txt)
    print("patch_preproc_arrays_free: the weight preprocessor now frees its four per-row arrays "
          "(" + SRC + ")")


if __name__ == "__main__":
    main()
