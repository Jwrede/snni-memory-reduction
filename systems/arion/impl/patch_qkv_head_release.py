#!/usr/bin/env python3
"""ARION a5_qkv_head_release: drop each head's Q, K, V ciphertext references after that head's
attention (ComputeMultiHeadAttentionMT1). SplitCiphertextMatricesByHeads returns sub-slices, so the
whole Q/K/V arrays stay reachable until the head loop returns; head h's columns are unread after
iteration h. Donor: MOAI-CPU m3_x_copy_release / m4_enc_ecd_x_release. Runs after
patch_activation_input_release.py.
Usage: python3 patch_qkv_head_release.py <path to the arion source tree>
"""
import os
import re
import sys

SRC = "pkg/bert/attentionMT.go"

ANCHOR = """		ctAttentionHeads[headIdx] = ctAttention
	}
"""

PATCH = """		ctAttentionHeads[headIdx] = ctAttention
		// a5_qkv_head_release: this head's Q, K and V columns are dead once its attention is
		// computed. The split slices share the backing arrays of ctQ/ctK/ctV.Ciphertexts, so
		// clearing them here releases the columns as the loop advances instead of at return.
		for i := range ctQsplit[headIdx].Ciphertexts {
			ctQsplit[headIdx].Ciphertexts[i] = nil
		}
		for i := range ctKsplit[headIdx].Ciphertexts {
			ctKsplit[headIdx].Ciphertexts[i] = nil
		}
		for i := range ctVsplit[headIdx].Ciphertexts {
			ctVsplit[headIdx].Ciphertexts[i] = nil
		}
	}
"""

MARKER_ANCHOR = """	ctAttentionHeads := make([]*he.CiphertextMatrices, modelParams.NumHeads)
"""

MARKER = """	ctAttentionHeads := make([]*he.CiphertextMatrices, modelParams.NumHeads)
	fmt.Fprintln(os.Stderr, "LEVER|qkv_head_release|armed")
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "a5_qkv_head_release" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    # Both edits are confined to ComputeMultiHeadAttentionMT1, the function menu 12 calls.
    fn_start = txt.find("func ComputeMultiHeadAttentionMT1(")
    if fn_start < 0:
        sys.exit("FAILED: no ComputeMultiHeadAttentionMT1")
    fn_end = txt.index("\nfunc ", fn_start + 1)
    body = txt[fn_start:fn_end]
    if body.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the head-loop store matched {body.count(ANCHOR)} times, expected exactly 1")
    if body.count(MARKER_ANCHOR) != 1:
        sys.exit(f"FAILED: the heads slice matched {body.count(MARKER_ANCHOR)} times, expected 1")

    # Refuse if the split slices or ctQ/ctK/ctV are read after the head loop (comments stripped).
    tail = body[body.index(ANCHOR) + len(ANCHOR):]
    tail = re.sub(r"//[^\n]*", "", tail)
    uses = re.findall(r"\bct[QKV](split)?\b", tail)
    if uses:
        sys.exit(f"FAILED: Q/K/V are read {len(uses)} more time(s) after the head loop; clearing "
                 "them there would lose something still in use")

    body = body.replace(ANCHOR, PATCH, 1)
    body = body.replace(MARKER_ANCHOR, MARKER, 1)
    txt = txt[:fn_start] + body + txt[fn_end:]
    if '"os"' not in txt:
        txt = txt.replace('\t"math"\n', '\t"math"\n\t"os"\n', 1)
    if '"os"' not in txt:
        sys.exit("FAILED: could not add the os import")

    open(src, "w").write(txt)
    print("patch_qkv_head_release: each head's Q/K/V columns are released after that head")


if __name__ == "__main__":
    main()
