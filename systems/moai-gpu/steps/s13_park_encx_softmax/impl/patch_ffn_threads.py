#!/usr/bin/env python3
"""MOAI-GPU s4: FFN/GELU thread count 8 -> 2 (the after_gelu reservation gap is parallel workspace).
Paid.

    patch_ffn_threads.py <moai source tree>

At after_gelu the pool reserves 8,749.9 MiB above live data: per-thread CUDA-stream workspace.
Expected ~-5.8% of the 108,384 MiB peak; runtime cost measured. Gate T1s, byte-identical.
"""
import os
import re
import sys

SRC = "src/include/test/test_single_layer.cuh"

ANCHOR = "    const int nthreads = std::max(1, std::min(max_threads, 32));\n"

PATCH = """    // s4_ffn_threads: this stage's peak is the PARALLEL WORKSPACE, not the data. At `after_gelu`
    // the pool holds 99,634.1 MiB live and has reserved 108,384 -- a gap of 8,749.9 MiB that is no
    // object, and each worker here takes its own CUDA stream and its own per-ciphertext
    // temporaries. Fewer workers, less workspace. PAID: this costs runtime, and how much is what
    // the run measures.
    const int nthreads = std::max(1, std::min(max_threads, 2));
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "s4_ffn_threads" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the thread cap matched {txt.count(ANCHOR)} times, expected exactly 1")

    # Must be the FFN/GELU cap (between before_intermediate and after_gelu), not another; this file
    # caps threads in more than one place.
    at = txt.index(ANCHOR)
    before = txt[:at]
    after = txt[at:]
    if "before_intermediate" not in before:
        sys.exit("FAILED: the matched cap sits BEFORE the `before_intermediate` marker, so it is "
                 "not the FFN/GELU stage this step is derived from")
    if "after_gelu" not in after:
        sys.exit("FAILED: no `after_gelu` marker after the matched cap; the stage this step is "
                 "derived from does not end where the derivation assumes")
    # And nothing may already have lowered it: the derivation's arithmetic assumes 8 live workers.
    m = re.search(r"const int nthreads = std::max\(1, std::min\(max_threads, (\d+)\)\);", before)
    if m:
        sys.exit(f"FAILED: an earlier cap of {m.group(1)} already applies above this stage; the "
                 "8-worker baseline this step predicts against is not what would run")

    txt = txt.replace(ANCHOR, PATCH, 1)
    open(src, "w").write(txt)
    print("patch_ffn_threads: FFN/GELU workers capped at 2 (" + SRC + ")")


if __name__ == "__main__":
    main()
