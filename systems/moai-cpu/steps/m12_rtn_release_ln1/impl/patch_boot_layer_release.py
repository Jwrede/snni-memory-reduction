#!/usr/bin/env python3
"""MOAI-CPU m5_boot_layer_release: release boot_layer after the second residual add; dead from there and never reused, so one edit. FREE. Base = m4_enc_ecd_x_release.

    patch_boot_layer_release.py <moai source tree>
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

# boot_layer's last read is the second residual (rtn2 += boot_layer). It is never reused, and the
# peak (after_bootstrap4) is later. One edit: swap it right after the residual loop closes.
ANCHOR = "        evaluator.add_inplace(rtn2[i],boot_layer[i]);\n    }\n"
PATCH = """        evaluator.add_inplace(rtn2[i],boot_layer[i]);
    }
    // LEVER boot_layer_release: the bootstrapped residual is dead after this second residual add and
    // never reused; the peak (after_bootstrap4) is later. No resize needed (unlike x_copy/enc_ecd_x).
    {
        size_t _snni_released = boot_layer.size();
        vector<Ciphertext>().swap(boot_layer);
        fprintf(stderr, "LEVER|boot_layer_release|released=%zu\\n", _snni_released);
        fflush(stderr);
    }
"""

LAST_READ = "evaluator.add_inplace(rtn2[i],boot_layer[i]);"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "boot_layer_release" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if "vector<Ciphertext>().swap(boot_layer);" in txt:
        sys.exit("FAILED: this tree already swaps boot_layer")
    if txt.count(LAST_READ) != 1:
        sys.exit(f"FAILED: boot_layer's residual add matched {txt.count(LAST_READ)} times, expected 1")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the residual-loop anchor matched {txt.count(ANCHOR)} times, expected 1")
    # boot_layer must not be used after the release point.
    after = txt[txt.index(ANCHOR) + len(ANCHOR):]
    if "boot_layer" in after:
        sys.exit("FAILED: boot_layer is used after the release point; a swap here is a use-after-free")

    txt = txt.replace(ANCHOR, PATCH, 1)
    open(src, "w").write(txt)
    out = open(src).read()
    if "LEVER|boot_layer_release" not in out or "swap(boot_layer);" not in out:
        sys.exit("FAILED: applied the patch but the marker or swap is absent")
    print("patch_boot_layer_release: boot_layer released after its last use (" + SRC + ")")


if __name__ == "__main__":
    main()
