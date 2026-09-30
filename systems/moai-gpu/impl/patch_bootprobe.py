#!/usr/bin/env python3
"""MOAI-GPU p_bootprobe: pool growth per bootstrap call or once (marker-only probe).

    patch_bootprobe.py <moai source tree>

One snni_mem_marker every 64 ciphertexts in the first bootstrap loop (12 over 768 calls). Linear in
i = per-call retention (lever); plateau = reserved once (allocator). No allocation, arithmetic or
ordering change.
"""
import os
import re
import sys

SRC = "src/include/test/test_single_layer.cuh"

# Anchored on s2_boot_input_release's own marker line (this probe builds on the s2 tree, which
# rewrote this loop). The pristine loop matched zero times.
ANCHOR ="""            if (i == 0 && j == 0) std::cerr << "LEVER|boot_input_release|sites=4" << std::endl;
"""

PATCH = """            if (i == 0 && j == 0) std::cerr << "LEVER|boot_input_release|sites=4" << std::endl;
            // p_bootprobe: the pool moves 768 x 34 MiB across this loop while the call is per
            // ciphertext. A marker every 64 calls separates "grows per call" (retention, a lever)
            // from "reserved once" (the allocator, already measured unfixable on this platform).
            if (((i*6+j) % 64) == 0) {
                char _snni_tag[48];
                snprintf(_snni_tag, sizeof(_snni_tag), "boot1_ct%d", i*6+j);
                snni_mem_marker(_snni_tag);
            }
"""

def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "p_bootprobe" in txt:
        sys.exit("FAILED: this tree already carries the probe")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the first bootstrap loop matched {txt.count(ANCHOR)} times, expected 1")

    # The marker mechanism must already exist: this probe's claim to be free is that it uses the
    # program's own instrumentation and adds none.
    inc = os.path.join(sys.argv[1], "src/include/include.cuh")
    if not os.path.exists(inc) or "snni_mem_marker" not in open(inc).read():
        sys.exit("FAILED: snni_mem_marker is not in include.cuh; this probe would be adding an "
                 "instrument rather than reading one")

    # Must be the FIRST bootstrap: the target 42,239.9 MiB figure was measured across
    # `after_selfoutput -> after_bootstrap1`.
    at = txt.index(ANCHOR)
    if "after_selfoutput" not in txt[:at][-6000:]:
        sys.exit("FAILED: the matched loop is not the one after `after_selfoutput`; the number this "
                 "probe explains belongs to the first bootstrap")

    txt = txt.replace(ANCHOR, PATCH, 1)
    if "#include <cstdio>" not in txt and "#include <stdio.h>" not in txt:
        txt = "#include <cstdio>\n" + txt
    open(src, "w").write(txt)
    print("patch_bootprobe: pool markers every 64 ciphertexts in the first bootstrap (" + SRC + ")")


if __name__ == "__main__":
    main()
