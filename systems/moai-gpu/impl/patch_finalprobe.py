#!/usr/bin/env python3
"""MOAI-GPU p_finalprobe: pool shape inside the final matmul (between after_gelu and after_final,
~33,800 MiB transient, unmarked).

    patch_finalprobe.py <moai source tree>

Marker every 8 of 128 outer iterations plus entry/exit. Linear in i = per-iteration plaintext churn
(lever); plateau = no lever. Marker-only.
"""
import os
import sys

SRC = "src/include/source/matrix_mul/Ct_pt_matrix_mul.cuh"

# The function is the anchor because the loop alone is not unique (this file has SIX identical
# `for (int i = 0; i < 128; ++i)` loops); the loop is then found INSIDE it.
FUNC =("vector<PhantomCiphertext> ct_pt_matrix_mul_wo_pre_w_mask_single"
        "(vector<PhantomCiphertext> &enc_X,")

LOOP = """#pragma omp for schedule(static)
    for (int i = 0; i < 128; ++i)
    {
"""

MARK = """#pragma omp for schedule(static)
    for (int i = 0; i < 128; ++i)
    {
      // p_finalprobe: the peak sits between after_gelu and after_final, and this call is the only
      // thing between them. A marker every 8 iterations separates "grows per iteration" (plaintext
      // churn on a pool that never returns memory, i.e. a lever) from "a plateau" (a genuine
      // simultaneous working set, i.e. no lever here).
      if ((i % 8) == 0) {
        char _snni_tag[48];
        snprintf(_snni_tag, sizeof(_snni_tag), "final_i%d", i);
        snni_mem_marker(_snni_tag);
      }
"""

ENTRY_ANCHOR = """  vector<PhantomCiphertext> output(col_W);
"""

ENTRY = """  vector<PhantomCiphertext> output(col_W);
  snni_mem_marker("final_entry");
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "p_finalprobe" in txt:
        sys.exit("FAILED: this tree already carries the probe")
    if txt.count(FUNC) != 1:
        sys.exit(f"FAILED: the function signature matched {txt.count(FUNC)} times, expected 1")

    at = txt.index(FUNC)

    # The marker mechanism must already exist: this probe's claim to be free is that it uses the
    # program's own instrumentation and adds none.
    inc = os.path.join(sys.argv[1], "src/include/include.cuh")
    if not os.path.exists(inc) or "snni_mem_marker" not in open(inc).read():
        sys.exit("FAILED: snni_mem_marker is not in include.cuh; this probe would be adding an "
                 "instrument rather than reading one")
    if '#include "include.cuh"' not in txt[:at]:
        sys.exit("FAILED: this file does not include include.cuh, so the marker is not in scope")

    # Entry marker placed on the function's own first statement so it cannot land in a neighbour.
    ent = txt.find(ENTRY_ANCHOR, at)
    if ent < 0 or ent - at > 1200:
        sys.exit("FAILED: `vector<PhantomCiphertext> output(col_W);` is not within this function's "
                 "opening; the probe would be marking something else")

    # The loop found INSIDE the function; the distance bound makes that checkable (a far match means
    # the function's body was not what was matched).
    lp = txt.find(LOOP, at)
    if lp < 0:
        sys.exit("FAILED: the 128-iteration loop was not found after the function signature")
    if lp - at > 6000:
        sys.exit(f"FAILED: the nearest 128-loop is {lp - at} chars past the signature, too far to "
                 "be this function's own body")

    # Patch the LOOP first, because patching the entry first would shift `lp`.
    txt = txt[:lp] + MARK + txt[lp + len(LOOP):]
    txt = txt[:ent] + ENTRY + txt[ent + len(ENTRY_ANCHOR):]

    if "#include <cstdio>" not in txt and "#include <stdio.h>" not in txt:
        txt = "#include <cstdio>\n" + txt
    open(src, "w").write(txt)
    print("patch_finalprobe: pool markers every 8 iterations of the final matmul (" + SRC + ")")


if __name__ == "__main__":
    main()
