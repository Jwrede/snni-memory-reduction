#!/usr/bin/env python3
"""MOAI-CPU att_keys_late_create: generate the 14 attention rotation keys right before the layer
loop instead of at key generation (not resident during input preparation).

    patch_att_keys_late_create.py <moai source tree>

Basis: highest window = input preparation before layer 0 (52.6 GiB): attention keys (17.2 GB, first
read in layer 0's attention), input at bootstrap level (16.9 GB), residual copy (16.9 GB, materialised
to be parked). Keys created after the copy is parked. Same key bytes (seeded PRNG factory, one fresh
stream per call; call position irrelevant). Free.
Requires m1's minimal key set and the copy_after_switch park (anchors).
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

CREATE_ANCHOR = "    keygen.create_galois_keys(att_steps, gal_keys);\n"
CREATE = "    // att_keys_late_create: created right before the layer loop (see below).\n"

RUN_BEGIN_ANCHOR = 'snni_mark("run_begin");\n'
RUN_BEGIN = """// att_keys_late_create: the input is prepared and its copy parked; only now are the attention
// keys generated, so they are not resident during the input preparation window.
keygen.create_galois_keys(att_steps, gal_keys);
fprintf(stderr, "LEVER|att_keys_late_create|n=%zu\\n", att_steps.size());
fflush(stderr);
snni_mark("att_keys_created");
""" + RUN_BEGIN_ANCHOR


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "att_keys_late_create" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, anchor in (("create", CREATE_ANCHOR), ("run_begin", RUN_BEGIN_ANCHOR)):
        n = txt.count(anchor)
        if n != 1:
            sys.exit(f"FAILED: the {name} anchor appears {n} times, expected 1")
    # the keys must not be read between the old and the new creation point
    a = txt.index(CREATE_ANCHOR)
    b = txt.index(RUN_BEGIN_ANCHOR)
    between = txt[a + len(CREATE_ANCHOR):b]
    import re
    uses = [m.group(0) for m in re.finditer(r"\bgal_keys\b(?!_boot)", between)]
    if uses:
        sys.exit(f"FAILED: gal_keys is used {len(uses)} time(s) between keygen and run_begin")
    txt = txt.replace(CREATE_ANCHOR, CREATE, 1)
    txt = txt.replace(RUN_BEGIN_ANCHOR, RUN_BEGIN, 1)
    open(src, "w").write(txt)
    out = open(src).read()
    if out.count("keygen.create_galois_keys(att_steps, gal_keys);") != 1:
        sys.exit("FAILED: post-check")
    print("OK: att_keys_late_create applied to " + src)


if __name__ == "__main__":
    main()
