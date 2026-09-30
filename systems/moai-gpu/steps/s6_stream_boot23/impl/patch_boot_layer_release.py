#!/usr/bin/env python3
"""MOAI-GPU s3: release each boot_layer element after its last reader (the add at line 1242).

    patch_boot_layer_release.py <moai source tree>

Target: layernorm2 reserves 14,496 MiB (11.8% of the peak) to hold 768 MiB more (worst reuse
ratio). Loop serial (omp pragma commented out). Free (dead-value release). Expected up to -11.8%,
less if mod_switch_to_inplace at 1241 already frees. Gate T1s.
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

    # Nothing may read boot_layer after this loop; comments stripped first so this step's own
    # comment cannot satisfy the check.
    tail = re.sub(r"//[^\n]*", "", txt[at + len(ANCHOR):])
    uses = re.findall(r"boot_layer\s*\[", tail)
    if uses:
        sys.exit(f"FAILED: `boot_layer[...]` is read {len(uses)} more time(s) after the consume "
                 "loop; releasing it there would lose something still in use")

    # And the loop must be serial: a live `#pragma omp parallel for` over the same index would need
    # a single-writer argument this step does not make (stock has that pragma commented out).
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
