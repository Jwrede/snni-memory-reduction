#!/usr/bin/env python3
"""MOAI-GPU: inserts the borrowed park/disk-bank helpers into include.cuh (infrastructure, not a step).

    patch_park_helpers.py <moai source tree>

Adds offload_cipher_to_host, reload_cipher_from_host and the write-through DiskBank; no call site.
Block = impl/borrowed/wpark_helpers.cuh (retired w1_park_residuals image, include.cuh:227-454).
Without MOAI_SPILL_DIR: park in host RAM; with it: disk bank.
"""
import os
import sys

INC = "src/include/include.cuh"
ANCHOR = "// source code\n"
HELPERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "borrowed", "wpark_helpers.cuh")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    inc = os.path.join(sys.argv[1], INC)
    if not os.path.exists(inc):
        sys.exit("FAILED: no " + INC + " under " + sys.argv[1])
    if not os.path.exists(HELPERS):
        sys.exit("FAILED: no borrowed/wpark_helpers.cuh beside this patch")

    txt = open(inc).read()
    if "offload_vector_to_host" in txt:
        sys.exit("FAILED: this tree already carries the park helpers")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the `// source code` marker matched {txt.count(ANCHOR)} times, expected 1")

    helpers = open(HELPERS).read()
    for need in ("offload_cipher_to_host", "reload_cipher_from_host", "MOAI_SPILL_DIR"):
        if need not in helpers:
            sys.exit(f"FAILED: borrowed/wpark_helpers.cuh does not contain `{need}`")

    # The helpers must precede the source includes; the headers below are compiled against them.
    if "#include <fcntl.h>" not in txt:
        if "#include <unistd.h>" not in txt:
            sys.exit("FAILED: include.cuh has neither fcntl.h nor unistd.h; the bank helpers need "
                     "open/pread/pwrite and this is not the tree they were taken from")
        txt = txt.replace("#include <unistd.h>", "#include <unistd.h>\n#include <fcntl.h>", 1)

    txt = txt.replace(ANCHOR,
                      "// ---- borrowed verbatim from the retired w1_park_residuals image "
                      "(include.cuh:227-454) ----\n" + helpers +
                      "// ---- end borrowed block ----\n\n" + ANCHOR, 1)
    open(inc, "w").write(txt)
    print("patch_park_helpers: %d borrowed lines inserted into %s"
          % (len(helpers.splitlines()), INC))


if __name__ == "__main__":
    main()
