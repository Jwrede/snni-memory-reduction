#!/usr/bin/env python3
"""MOAI-CPU m17_rtn_release_ln1: release bootstrap1's output after layernorm1 consumed it.

    patch_rtn_release_ln1.py <moai source tree>

rtn (768 cts, 22 MB each, 16.9 GB) gets the residual add, last read by layernorm1; bootstrap2 rewrites
it only at its end. m16 line peak = bootstrap2 in-phase 82.2 GiB, rank 2 = this array (rank 1: live
phase keys). Bit-identical, free.
Where: after the after_layernorm1 marker; nothing reads rtn until stage_bootstrap_chunked(bootstrapper,
rtn, layernorm_selfoutput, 768, "bootstrap2") resizes it. Later bootstraps already free their input
(swap(rtn) after the FFN) or use rtn2.
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"
ANCHOR = '    snni_mark("after_layernorm1");\n'
ADDED = ANCHOR + """    // m17_rtn_release_ln1: bootstrap1's output is dead from here (layernorm1 was its last reader);
    // bootstrap2 resizes and refills `rtn` only after its last chunk.
    {
        size_t _snni_n = rtn.size();
        vector<Ciphertext>().swap(rtn);
        malloc_trim(0);
        fprintf(stderr, "LEVER|rtn_release_ln1|released=%zu\\n", _snni_n);
        fflush(stderr);
    }
"""
BOOT2 = 'stage_bootstrap_chunked(bootstrapper, rtn, layernorm_selfoutput, 768, "bootstrap2");'


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "rtn_release_ln1" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the after_layernorm1 marker appears {txt.count(ANCHOR)} times, expected 1")
    if txt.count(BOOT2) != 1:
        sys.exit("FAILED: bootstrap2 call not found once; the dead window this lever relies on is not the expected one")
    # The window must be free of rtn reads: between the marker and the bootstrap2 call.
    a = txt.index(ANCHOR) + len(ANCHOR)
    b = txt.index(BOOT2)
    if a >= b:
        sys.exit("FAILED: after_layernorm1 marker is not before the bootstrap2 call")
    window = txt[a:b]
    reads = [ln for ln in window.splitlines() if "rtn" in ln and not ln.strip().startswith("//")]
    if reads:
        sys.exit("FAILED: rtn is referenced between layernorm1 and bootstrap2:\n" + "\n".join(reads))
    txt = txt.replace(ANCHOR, ADDED, 1)
    open(src, "w").write(txt)
    if "rtn_release_ln1" not in open(src).read():
        sys.exit("FAILED: applied but marker absent")
    print("patch_rtn_release_ln1: rtn released after layernorm1 (" + SRC + ")")


if __name__ == "__main__":
    main()
