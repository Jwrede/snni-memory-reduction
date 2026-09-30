#!/usr/bin/env python3
"""MOAI-GPU s4: FFN/GELU thread count 8 -> 2 (the after_gelu reservation gap is parallel workspace).
Paid.

    patch_ffn_threads.py <moai source tree>

Basis: s3_boot_layer_release reservation series:
    phase                 used       reserved
    -> after_intermediate +27,648     +3,552
    -> after_gelu          +6,144     +8,416     <- reserves MORE than the data it holds
    -> after_final / bootstrap3 / layernorm2 / bootstrap4:  0
at after_gelu: 99,634.1 MiB live, 108,384 reserved (gap 8,749.9 MiB, no object). FFN matmul and GELU:
    const int nthreads = std::max(1, std::min(max_threads, 32));      // = 8 here, OMP_NUM_THREADS=8
each thread with its own CUDA stream and per-ciphertext temporaries.
Context: three levers at before_attention (49.6%) null (s4_input_copy_fused, s5_park_input_copy,
s6_fused_park_input); s2 (-9.0%) and s3 (-10.4%) act at or after the last growth event.
Paid: fewer workers (runtime). Value 2 from retired w5 (its numbers not cited as evidence).
Expected: after_gelu +8,416 -> ~+2,100 (-6,300 MiB, -5.8% of 108,384 MiB; spread 2.94%). No scaling
with threads = not workspace.
Gate T1s, byte-identical.

Usage: python3 patch_ffn_threads.py <path to the moai source tree>
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

    # IT MUST BE THE FFN/GELU STAGE'S CAP AND NOT ANOTHER, checked against what stands around it
    # rather than by line number: this file caps threads in more than one place and the one that
    # matters is the one between `before_intermediate` and `after_gelu`.
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
