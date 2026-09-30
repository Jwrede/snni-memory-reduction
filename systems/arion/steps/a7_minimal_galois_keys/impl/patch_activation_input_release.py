#!/usr/bin/env python3
"""ARION a4: drop each activation input reference after MulNew reads it (ctMats.Ciphertexts[i]=nil).
Basis: a3_threads16 rank 2 (matmul output = activation input); stock keeps input and output arrays
live together. Free (dead after last use); all five callers checked for re-reads.
Usage: python3 patch_activation_input_release.py <path to the arion source tree>
"""
import os
import re
import sys

SRC = "pkg/math/activation/polynomialMT.go"

ANCHOR = """				res, err := localEval.MulNew(ctMats.Ciphertexts[i], scalarmul)
				if err != nil {
					panic(err)
				}
"""

PATCH = """				res, err := localEval.MulNew(ctMats.Ciphertexts[i], scalarmul)
				if err != nil {
					panic(err)
				}
				// a4_activation_input_release: the input element is dead the instant MulNew has
				// read it. The stock code keeps the WHOLE input array resident until the call
				// returns, so by the end both the complete input and the complete output are live.
				// Dropping the reference here lets the collector take each input as the loop
				// advances. Every caller assigns its input on the line before the call and never
				// mentions it again, checked at all five sites in pkg/bert/bertMT.go.
				ctMats.Ciphertexts[i] = nil
"""

MARKER_ANCHOR = """	var wg sync.WaitGroup
"""

MARKER = """	fmt.Fprintln(os.Stderr, "LEVER|activation_input_release|armed")
	var wg sync.WaitGroup
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "a4_activation_input_release" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the MulNew read matched {txt.count(ANCHOR)} times, expected exactly 1")
    if txt.count(MARKER_ANCHOR) != 1:
        sys.exit(f"FAILED: the WaitGroup matched {txt.count(MARKER_ANCHOR)} times, expected 1")

    # Refuse if ctMats.Ciphertexts is read again after MulNew (its asserted last reader); comments stripped first.
    tail = txt[txt.index(ANCHOR) + len(ANCHOR):]
    tail = re.sub(r"//[^\n]*", "", tail)
    uses = re.findall(r"ctMats\.Ciphertexts\s*\[", tail)
    if uses:
        sys.exit(f"FAILED: `ctMats.Ciphertexts[...]` is read {len(uses)} more time(s) after the "
                 "MulNew; dropping the reference there would lose something still in use")

    txt = txt.replace(ANCHOR, PATCH, 1)
    txt = txt.replace(MARKER_ANCHOR, MARKER, 1)

    # Add fmt/os to the existing import block (not a second block, so gofmt stays quiet).
    if '"fmt"' not in txt:
        txt = txt.replace('import (\n\t"Arion/pkg/he"', 'import (\n\t"Arion/pkg/he"\n\t"fmt"', 1)
    if '"os"' not in txt:
        txt = txt.replace('import (\n\t"Arion/pkg/he"', 'import (\n\t"Arion/pkg/he"\n\t"os"', 1)

    open(src, "w").write(txt)
    print("patch_activation_input_release: the activation releases each input as it consumes it")


if __name__ == "__main__":
    main()
