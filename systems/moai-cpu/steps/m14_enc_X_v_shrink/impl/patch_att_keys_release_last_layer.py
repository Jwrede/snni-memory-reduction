#!/usr/bin/env python3
"""MOAI-CPU att_keys_release_last_layer: drop the attention rotation keys after the last layer's
attention block (no later reader).

    patch_att_keys_release_last_layer.py <moai source tree>

gal_keys (14 keys, 17.2 GB) is read only by the attention block; park_att_keys parks it only if
another layer follows, so in the last layer it stays through four bootstraps, FFN and both
layernorms. m16c peak (97.7 GiB) = last layer's bootstrap2, rank 2 = gal_keys 17.2 GB (dead). Free
(no write), bit-identical.
Where: after the after_attention marker, guarded on layer_id + 1 >= snni_n_layers (complement of
park_att_keys' guard).
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"
ANCHOR = '    snni_mark("after_attention");\n'
ADDED = ANCHOR + """    // att_keys_release_last_layer: in the final layer nothing reads `gal_keys` after the attention
    // block (park_att_keys covers the layers that are followed by another one).
    if (layer_id + 1 >= snni_n_layers) {
        gal_keys = GaloisKeys();
        malloc_trim(0);
        fprintf(stderr, "LEVER|att_keys_release_last_layer|layer=%d|action=released\\n", layer_id);
        fflush(stderr);
        snni_mark("att_keys_released");
    }
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "att_keys_release_last_layer" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the after_attention marker appears {txt.count(ANCHOR)} times, expected 1")
    if "int snni_n_layers = snni_int_env(" not in txt:
        sys.exit("FAILED: no snni_n_layers in the tree")
    txt = txt.replace(ANCHOR, ADDED, 1)
    # malloc_trim needs <malloc.h>; trees before the phase split (m0..m6) do not include it yet.
    if "#include <malloc.h>" not in txt:
        txt = "#include <malloc.h>\n" + txt
    open(src, "w").write(txt)
    if "att_keys_release_last_layer" not in open(src).read():
        sys.exit("FAILED: applied but marker absent")
    print("patch_att_keys_release_last_layer: gal_keys released after the last attention block (" + SRC + ")")


if __name__ == "__main__":
    main()
