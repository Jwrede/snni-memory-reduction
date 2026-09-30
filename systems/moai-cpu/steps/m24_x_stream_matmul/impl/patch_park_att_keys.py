#!/usr/bin/env python3
"""MOAI-CPU m1_park_att_keys: gal_keys (~43.9 GB, read only by the attention block) is written to disk after each attention phase and read back before the next, freeing blocks for bootstrap1. PAID (I/O).

    patch_park_att_keys.py <moai source tree>
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

# Both anchors are the campaign's own phase markers (inserted by pin_seed_and_gate.py, which runs
# first), so this patch fails loudly if the marker section is ever removed.
ANCHOR_PARK = '    snni_mark("after_attention");'
ANCHOR_LOAD = '    snni_mark("before_attention");'

HELPER = """// ---------------------------------------------------------------------------------------
// m1_park_att_keys: where the parked attention key set lives. The run directory by default, which
// is the results directory the sbatch cds into; MOAI_PARK_DIR overrides it. The NAME matters:
// `harvest_step.sh` excludes `key_*.dat`, and this file is 43.9 GB.
static std::string snni_att_key_path() {
    const char *d = getenv("MOAI_PARK_DIR");
    std::string dir = (d && *d) ? std::string(d) : std::string(".");
    return dir + "/key_gal_parked.dat";
}
// ---------------------------------------------------------------------------------------

void all_layer_test(){"""

PARK = ANCHOR_PARK + """
    // m1_park_att_keys: `gal_keys` is read by single_att_block and by nothing else. The attention
    // block is over; the four bootstraps, the layernorms, the FFN and the two matmuls that follow
    // never touch it. Park it so bootstrap1's +85.9 GB can be served from the freed blocks.
    if (layer_id + 1 < snni_n_layers) {
        {
            std::ofstream _snni_pk(snni_att_key_path(), std::ios::binary);
            if (!_snni_pk) {
                fprintf(stderr, "LEVER|park_att_keys|FAILED|cannot open %s\\n",
                        snni_att_key_path().c_str());
                fflush(stderr);
                exit(9);
            }
            gal_keys.save(_snni_pk, compr_mode_type::none);
        }
        gal_keys = GaloisKeys();
        fprintf(stderr, "LEVER|park_att_keys|layer=%d|action=parked|path=%s\\n",
                layer_id, snni_att_key_path().c_str());
        fflush(stderr);
        snni_mark("att_keys_parked");
    }"""

LOAD = """    // m1_park_att_keys: the next layer's attention block needs the set again. Read back rather
    // than regenerate: the same bytes, so the seeded PRNG stream is untouched and the gate compares
    // like with like.
    if (layer_id > 0 && gal_keys.data().empty()) {
        std::ifstream _snni_pk(snni_att_key_path(), std::ios::binary);
        if (!_snni_pk) {
            fprintf(stderr, "LEVER|park_att_keys|FAILED|cannot reopen %s\\n",
                    snni_att_key_path().c_str());
            fflush(stderr);
            exit(9);
        }
        gal_keys.load(context, _snni_pk);
        fprintf(stderr, "LEVER|park_att_keys|layer=%d|action=reloaded\\n", layer_id);
        fflush(stderr);
        snni_mark("att_keys_reloaded");
    }
""" + ANCHOR_LOAD


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "m1_park_att_keys" in txt:
        sys.exit("FAILED: this tree already carries the lever")

    # The anchor markers must each appear exactly once; a tree without them never ran
    # pin_seed_and_gate.py and would place the park wrong with no symptom until the numbers differ.
    for name, anchor in (("park", ANCHOR_PARK), ("reload", ANCHOR_LOAD)):
        n = txt.count(anchor)
        if n != 1:
            sys.exit(f"FAILED: the {name} anchor {anchor.strip()!r} appears {n} times, expected 1. "
                     f"Run pin_seed_and_gate.py first, or re-derive this patch.")

    # `snni_n_layers` bounds the layer loop; without it the park could write 43.9 GB nobody reads.
    if "int snni_n_layers = snni_int_env(" not in txt:
        sys.exit("FAILED: no `snni_n_layers` in the tree; the depth knob is what tells this lever "
                 "whether another layer follows")

    if "void all_layer_test(){" not in txt:
        sys.exit("FAILED: cannot find all_layer_test to place the path helper above")
    txt = txt.replace("void all_layer_test(){", HELPER, 1)

    txt = txt.replace(ANCHOR_PARK, PARK, 1)
    txt = txt.replace(ANCHOR_LOAD, LOAD, 1)

    open(src, "w").write(txt)
    for what in ("snni_att_key_path", "action=parked", "action=reloaded"):
        if what not in open(src).read():
            sys.exit(f"FAILED: applied the patch but {what!r} is absent from {SRC}")
    print("patch_park_att_keys: gal_keys parked between attention blocks (" + SRC + ")")


if __name__ == "__main__":
    main()
