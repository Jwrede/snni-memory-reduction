#!/usr/bin/env python3
"""MOAI-GPU s3: release each boot_layer element after its last reader.

    patch_boot_layer_release.py <moai source tree>

Rule: on a pool that never returns memory, a lever must hand freed blocks to a later allocation
(null: off-path-runs/s3_inter_output_free/, s3_input_copy_fused/, MOAI-CPU m1_boot_out_move +4.36%;
worked: s2_boot_input_release, 768 MB released, pool 26,879.9 MB lower).
Target, from p_finalprobe (reserved per live MiB):
    phase                 used       reserved    ratio
    -> before_attention  +27,648     +53,792      1.95
    -> after_bootstrap3  +15,360        +352      0.02   s2's lever, near-perfect reuse
    -> after_layernorm2     +768     +14,496     18.90   <- THIS STEP
14,496 MiB = 11.8% of the peak (variation >= 1.08% proven, 2.94% observed; off-path.md).
boot_layer (16,911.4 MB, 15.2%, 768 allocations), test_single_layer.cuh:
    853   vector<PhantomCiphertext> boot_layer(layernorm_selfoutput_size);
    868   boot_layer[i*6+j] = rtn[i*6+j];                       filled
    885   ... chain_depth() print
    1241  evaluator.mod_switch_to_inplace(boot_layer[i], rtn2[i].params_id());
    1242  evaluator.add_inplace(rtn2[i], boot_layer[i]);        LAST reader
    (never mentioned again)
loop 1240-1243 serial (omp pragma commented out). Element released after line 1242.
Free (dead after last use).
Open: mod_switch_to_inplace at 1241 may already return the depth-20 buffer; either outcome is
informative.
Expected: up to -14,496 MiB (~-11.8%).
Falsifier: gate T1s byte-identical; a fault means 1242 is not the last reader.

Usage: python3 patch_boot_layer_release.py <path to the moai source tree>
"""
import os
import re
import sys

SRC = "src/include/test/test_single_layer.cuh"

ANCHOR = """    for (int i = 0; i < num_col; ++i){
        evaluator.mod_switch_to_inplace(boot_layer[i], rtn2[i].params_id());
        evaluator.add_inplace(rtn2[i],boot_layer[i]);
    }
"""

PATCH = """    for (int i = 0; i < num_col; ++i){
        evaluator.mod_switch_to_inplace(boot_layer[i], rtn2[i].params_id());
        evaluator.add_inplace(rtn2[i],boot_layer[i]);
        // s3_boot_layer_release: this element's last reader is the add above, and the stock code
        // then holds the whole 768-wide array to the end of the layer -- through layernorm2, which
        // reserves 14,496 MiB to hold 768 MiB more because the pool has nothing of the right shape
        // to give it. Releasing here puts 768 blocks in the pool at the moment it starts asking.
        // The loop is serial, so there is no reader on another thread.
        boot_layer[i] = PhantomCiphertext();
    }
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "s3_boot_layer_release" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the consume loop matched {txt.count(ANCHOR)} times, expected exactly 1")

    at = txt.index(ANCHOR)

    # NOTHING MAY READ boot_layer AFTER THIS LOOP, which is the whole claim. Comments are stripped
    # first so this step's own comment cannot satisfy the check -- the trap that cost PUMA's v2
    # guard a build.
    tail = re.sub(r"//[^\n]*", "", txt[at + len(ANCHOR):])
    uses = re.findall(r"boot_layer\s*\[", tail)
    if uses:
        sys.exit(f"FAILED: `boot_layer[...]` is read {len(uses)} more time(s) after the consume "
                 "loop; releasing it there would lose something still in use")

    # AND THE LOOP MUST BE SERIAL. The release is per element and a live `#pragma omp parallel for`
    # over the same index would need the single-writer argument that this step explicitly does not
    # make. The stock source has that pragma commented out; if a predecessor step ever uncomments
    # it, this guard fails rather than letting the two changes silently combine.
    head = txt[:at]
    window = head[-400:]
    for line in window.splitlines():
        s = line.strip()
        if s.startswith("#pragma omp parallel for"):
            sys.exit("FAILED: a live `#pragma omp parallel for` sits above the consume loop; this "
                     "step's safety argument assumes the loop is serial")

    txt = txt.replace(ANCHOR, PATCH, 1)
    open(src, "w").write(txt)
    print("patch_boot_layer_release: boot_layer released as it is consumed (" + SRC + ")")


if __name__ == "__main__":
    main()
