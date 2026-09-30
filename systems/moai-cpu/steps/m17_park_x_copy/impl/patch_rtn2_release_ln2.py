#!/usr/bin/env python3
"""MOAI-CPU m18_rtn2_release_ln2: release bootstrap3's output after layernorm2 consumed it.

    patch_rtn2_release_ln2.py <moai source tree>

rtn2 (768 cts, 22 MB each, 16.9 GB) gets the residual add with boot_layer, last read by layernorm2;
bootstrap4 rewrites it only at its end (stage_bootstrap_chunked resizes and reloads after the last
chunk). m17 line peak = bootstrap4 in-phase 82.0 GiB, rank 2 = this array (rank 1: live phase keys).
Bit-identical, free.
Where: after the after_layernorm2 marker; the script checks no reference to rtn2 before
stage_bootstrap_chunked(bootstrapper, rtn2, layernorm_finaloutput, 768, "bootstrap4").
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"
ANCHOR = '    snni_mark("after_layernorm2");\n'
ADDED = ANCHOR + """    // m18_rtn2_release_ln2: bootstrap3's output is dead from here (layernorm2 was its last reader);
    // bootstrap4 resizes and refills `rtn2` only after its last chunk.
    {
        size_t _snni_n = rtn2.size();
        vector<Ciphertext>().swap(rtn2);
        malloc_trim(0);
        fprintf(stderr, "LEVER|rtn2_release_ln2|released=%zu\\n", _snni_n);
        fflush(stderr);
    }
"""
BOOT4 = 'stage_bootstrap_chunked(bootstrapper, rtn2, layernorm_finaloutput, 768, "bootstrap4");'


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "rtn2_release_ln2" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the after_layernorm2 marker appears {txt.count(ANCHOR)} times, expected 1")
    if txt.count(BOOT4) != 1:
        sys.exit("FAILED: bootstrap4 call not found once; the dead window this lever relies on is not the expected one")
    a = txt.index(ANCHOR) + len(ANCHOR)
    b = txt.index(BOOT4)
    if a >= b:
        sys.exit("FAILED: after_layernorm2 marker is not before the bootstrap4 call")
    window = txt[a:b]
    reads = [ln for ln in window.splitlines() if "rtn2" in ln and not ln.strip().startswith("//")]
    if reads:
        sys.exit("FAILED: rtn2 is referenced between layernorm2 and bootstrap4:\n" + "\n".join(reads))
    txt = txt.replace(ANCHOR, ADDED, 1)
    open(src, "w").write(txt)
    if "rtn2_release_ln2" not in open(src).read():
        sys.exit("FAILED: applied but marker absent")
    print("patch_rtn2_release_ln2: rtn2 released after layernorm2 (" + SRC + ")")


if __name__ == "__main__":
    main()
