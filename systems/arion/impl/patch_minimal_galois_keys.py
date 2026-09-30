#!/usr/bin/env python3
"""ARION minimal_galois_keys: generate only the rotation keys the BERT path uses.
GenerateKeysAndBtsKeysMT requests +/- i*NumBatch, i = 1..NumRow (256 galois elements, 76 MiB each,
19 GiB). Attention BSGS (ComputeAttentionMT3, the only rotating code on the menu-12 path) rotates by
i*NumBatch for 0 < i < BabyStep and by -/+ i*NumBatch*BabyStep for 0 < i < GiantStep. Bootstrapping
keys untouched. Donor: MOAI-CPU m2_minimal_att_keys. A missing key fails the rotation; T2 gate
compares every layer output with the baseline.
Usage: python3 patch_minimal_galois_keys.py <path to the arion source tree>
"""
import os
import sys

SRC = "pkg/he/keysMT.go"

FUNC = "func GenerateKeysAndBtsKeysMT("

ANCHOR = """	var rotNumbers []int
	for i := 1; i <= rangSlots; i++ {
		rotNumbers = append(rotNumbers, i*modelParams.NumBatch)
		rotNumbers = append(rotNumbers, -i*modelParams.NumBatch)
	}
"""

PATCH = """	var rotNumbers []int
	// minimal_galois_keys: only the rotations the attention BSGS performs (ComputeAttentionMT3):
	// baby steps i*NumBatch, giant steps -/+ i*NumBatch*BabyStep. Stock requested all +/- i*NumBatch
	// for i = 1..rangSlots.
	_ = rangSlots
	for i := 1; i < modelParams.BabyStep; i++ {
		rotNumbers = append(rotNumbers, i*modelParams.NumBatch)
	}
	for i := 1; i < modelParams.GiantStep; i++ {
		g := i * modelParams.NumBatch * modelParams.BabyStep
		rotNumbers = append(rotNumbers, -g, g)
	}
	fmt.Fprintf(os.Stderr, "LEVER|minimal_galois_keys|armed|rotations=%d\\n", len(rotNumbers))
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "minimal_galois_keys" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    start = txt.find(FUNC)
    if start < 0:
        sys.exit("FAILED: no " + FUNC)
    end = txt.find("\nfunc ", start + 1)
    body = txt[start:end]
    if body.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the rotation list matched {body.count(ANCHOR)} times in {FUNC}, expected 1")

    body = body.replace(ANCHOR, PATCH, 1)
    txt = txt[:start] + body + txt[end:]
    if '"os"' not in txt:
        txt = txt.replace('\t"fmt"\n', '\t"fmt"\n\t"os"\n', 1)
    if '"os"' not in txt:
        sys.exit("FAILED: could not add the os import")

    open(src, "w").write(txt)
    print("patch_minimal_galois_keys: only the BSGS rotation keys are generated")


if __name__ == "__main__":
    main()
