#!/usr/bin/env python3
"""SIGMA-GPU p_llamabuf_highwater probe: contents of the two 1 GiB pinned dealer buffers (2 GiB of a
21.05 GiB peak). No allocation or arithmetic change. llamaBuf1 reset per round (needs max), dummyBuf1
never reset (needs sum).

    patch_llamabuf_highwater.py <EzPC/GPU-MPC tree>

    SNNI_BUF|llamabuf_high=<max llamaKeySz>|llamabuf_alloc=1073741824
    SNNI_BUF|dummybuf_used=<dummyBuf2-dummyBuf1>|dummybuf_alloc=1073741824
"""
import os
import re
import sys

SRC = "backend/sigma.h"

DECL_ANCHOR = "    u8 *dummyBuf1, *dummyBuf2;\n"
DECL = """    u8 *dummyBuf1, *dummyBuf2;
    // p_llamabuf_highwater (PROBE): the largest single round seen in llamaBuf, which is what that
    // buffer must hold because it is drained and reset every round. dummyBuf needs no tracker: it
    // is never reset, so its final offset IS its high-water.
    size_t snniLlamaHigh = 0;
"""

TRACK_ANCHOR = "        size_t llamaKeySz = llamaBuf2 - llamaBuf1;\n"
TRACK = """        size_t llamaKeySz = llamaBuf2 - llamaBuf1;
        if (llamaKeySz > snniLlamaHigh) snniLlamaHigh = llamaKeySz;
"""

CLOSE_ANCHOR = """        /*size_t*/ keySize = keyBuf - startPtr;
"""

CLOSE = """        fprintf(stderr, "SNNI_BUF|llamabuf_high=%lu|llamabuf_alloc=%lu\\n",
                (unsigned long)snniLlamaHigh, (unsigned long)(1024UL * 1024 * 1024));
        fprintf(stderr, "SNNI_BUF|dummybuf_used=%lu|dummybuf_alloc=%lu\\n",
                (unsigned long)(dummyBuf2 - dummyBuf1), (unsigned long)(1024UL * 1024 * 1024));
        fflush(stderr);
        /*size_t*/ keySize = keyBuf - startPtr;
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "snniLlamaHigh" in txt:
        sys.exit("FAILED: this tree already carries the probe")
    for name, anchor, want in (("dummyBuf declaration", DECL_ANCHOR, 1),
                               ("llamaKeySz computation", TRACK_ANCHOR, 1),
                               ("keySize in close()", CLOSE_ANCHOR, 1)):
        if txt.count(anchor) != want:
            sys.exit(f"FAILED: the {name} matched {txt.count(anchor)} times, expected {want}")

    # dummyBuf2 MUST NEVER BE RESET: its final offset IS its high-water. Checked as exactly one
    # ASSIGNMENT (not mentions: line 302 names it twice per ternary, so a mention-count of 3 was wrong).
    assigns = len(re.findall(r"dummyBuf2\s*=(?!=)", txt))
    if assigns != 1:
        sys.exit(f"FAILED: `dummyBuf2` is assigned {assigns} times, expected exactly 1. If it is "
                 "reset anywhere, its final offset is NOT its high-water and this probe would "
                 "report a number that means something else")

    txt = txt.replace(DECL_ANCHOR, DECL, 1)
    txt = txt.replace(TRACK_ANCHOR, TRACK, 1)
    txt = txt.replace(CLOSE_ANCHOR, CLOSE, 1)
    open(src, "w").write(txt)
    print("patch_llamabuf_highwater: llamaBuf/dummyBuf high-water reported (" + SRC + ")")


if __name__ == "__main__":
    main()
