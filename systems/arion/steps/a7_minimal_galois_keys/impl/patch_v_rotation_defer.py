#!/usr/bin/env python3
"""ARION a7_v_rotation_defer: create each head's rotated V blocks at first use.
ComputeAttentionMT3 (pkg/bert/attentionMT.go) rotates Q, K, V for the whole head before Step 1;
rotated Q, K are read in Step 1 (QK^T), rotated V (12 baby-step rotations of 64 columns) only in
Step 5 (Exp * V). The V rotation moves to just before Steps 4/5: same rotations, later; output
unchanged, no added work. Donors: MOAI-CPU m26_att_keys_late_create, BumbleBee b3_weight_defer.
Runs after patch_minimal_galois_keys.py.
Usage: python3 patch_v_rotation_defer.py <path to the arion source tree>
"""
import os
import sys

SRC = "pkg/bert/attentionMT.go"
FUNC = "func ComputeAttentionMT3("
VROT = "\tctVRotated := matrix.RotateCiphertextMatricesHoistingMT(ctV, babySteps, eval, numThreads)\n"
STEP45 = "\t// Step4: Bootstrap + Inverse"
NEW = ("\t// a7_v_rotation_defer: the rotated V is read only in Step 5, so it is created here instead of\n"
       "\t// before Step 1 (same rotations of the same ciphertext, later).\n"
       "\tvRotDeferOnce.Do(func() { fmt.Fprintln(os.Stderr, \"LEVER|v_rotation_defer|armed\") })\n"
       + VROT)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    txt = open(src).read()
    if "a7_v_rotation_defer" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    start = txt.find(FUNC)
    if start < 0:
        sys.exit("FAILED: no " + FUNC)
    end = txt.find("\nfunc ", start + 1)
    body = txt[start:end]
    if body.count(VROT) != 1:
        sys.exit(f"FAILED: the V rotation matched {body.count(VROT)} times, expected 1")
    i45 = body.find(STEP45)
    if i45 < 0 or body.count(STEP45) != 1:
        sys.exit("FAILED: the Step 4/5 block is not unique in " + FUNC)
    ivr = body.index(VROT)
    # ctVRotated must not be read before the Step 4/5 block
    between = body[ivr + len(VROT):i45]
    if "ctVRotated" in between:
        sys.exit("FAILED: ctVRotated is read before Step 5; deferring it would break the code")
    body = body[:ivr] + body[ivr + len(VROT):]
    i45 = body.find(STEP45)
    ins = body.rfind("\n", 0, body.rfind("\n", 0, i45)) + 1   # before the separator line
    body = body[:ins] + NEW + "\n" + body[ins:]
    txt = txt[:start] + body + txt[end:]
    txt = txt.replace(FUNC, "var vRotDeferOnce sync.Once\n\n" + FUNC, 1)
    if '"os"' not in txt:
        txt = txt.replace('\t"fmt"\n', '\t"fmt"\n\t"os"\n', 1)
    for imp in ('"os"', '"sync"', '"fmt"'):
        if imp not in txt:
            sys.exit("FAILED: import " + imp + " missing")
    open(src, "w").write(txt)
    print("patch_v_rotation_defer: each head's V rotation is created just before Step 5")


if __name__ == "__main__":
    main()
