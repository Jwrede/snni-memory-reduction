#!/usr/bin/env python3
"""SIGMA-GPU: key buffer size settable via SIGMA_KEYBUF_MB (inert unless set), for geometry probes
(baseline 40960 MB, s5 17 GB, s6 1 GiB). Applied at the point of use after every model branch;
announces SNNI_KEYBUF|source=..|bytes=.. .

    patch_keybuf_env.py <sigma source tree>
"""
import os
import re
import sys

SRC = "experiments/sigma/sigma.cu"

ANCHOR = "    auto sigmaKeygen = new SIGMAKeygen<u64>(party, bw, scale, \"\", keyBufSz);"

OVERRIDE = """    // SNNI geometry probe: the key buffer, settable. INERT unless SIGMA_KEYBUF_MB is set, so a
    // binary carrying this line and run without the variable is the unpatched program. It sits at
    // the POINT OF USE, after every model branch, so it overrides whatever the chain last assigned
    // (20 GB at the baseline, 17 at s5, 1 GiB at s6) and one patch serves every probe.
    if (const char *snni_kb = getenv("SIGMA_KEYBUF_MB")) {
        keyBufSz = (u64)atoll(snni_kb) * 1024ULL * 1024ULL;
        printf("SNNI_KEYBUF|source=env|bytes=%lu\\n", (unsigned long)keyBufSz);
    } else {
        printf("SNNI_KEYBUF|source=default|bytes=%lu\\n", (unsigned long)keyBufSz);
    }
    fflush(stdout);
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "SNNI_KEYBUF|" in txt:
        sys.exit("FAILED: this tree already carries the override")

    # Anchor at the POINT OF USE (keygen constructor, after every model branch), not the assignment:
    # anchoring in the bert-base branch once collided with patch_dealer_window.py's two-line anchor
    # and broke a LATER patch in the cumulative chain. This line is touched by no other patch here.
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the keygen construction matched {txt.count(ANCHOR)} times, expected 1")
    txt = txt.replace(ANCHOR, OVERRIDE + ANCHOR, 1)

    # getenv/atoll (<cstdlib>) and printf (<cstdio>) are already used in this file, so the override
    # adds no headers; checked because a missing declaration would fail the build late in the chain.
    if "atoi(" not in txt or "printf(" not in txt:
        sys.exit("FAILED: this file does not already use atoi/printf, so the override's headers "
                 "are not guaranteed to be in scope")

    open(src, "w").write(txt)
    if "SNNI_KEYBUF|source=env" not in open(src).read():
        sys.exit("FAILED: applied the patch but its marker is absent")
    print("patch_keybuf_env: key buffer settable via SIGMA_KEYBUF_MB, default unchanged (" + SRC + ")")


if __name__ == "__main__":
    main()
