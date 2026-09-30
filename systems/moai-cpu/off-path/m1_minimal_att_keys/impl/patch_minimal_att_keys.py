#!/usr/bin/env python3
"""MOAI-CPU m1: 17 of the 31 attention Galois keys are never requested. Free.

    patch_minimal_att_keys.py <moai source tree>

    keygen.create_galois_keys(gal_keys);            ->   keygen.create_galois_keys(att_steps, gal_keys);

Object: m0_default depth-7 table:
    103,733.8 MB   25.3%   seal::util::encrypt_zero_symmetric  <- MemoryPoolHeadMT::get()
     35,239.2 MB    8.6%   seal::Evaluator::mod_switch_scale_to_next
     32,013.5 MB    7.8%   Ct_pt_matrix_mul.hpp:87 (the largest PROGRAM-owned row)
encrypt_zero_symmetric = key-switching key components. Poller trace: setup ramp +43.9 GB over 150 s,
then +58.7 GB over 235 s; 43.9 + 58.7 = 102.6 GB vs 103.7 (two Galois sets; gal_keys ~1.42 GB x 31).

Proof from this tree:
  1. gal_keys is read at four rotation sites, all in Ct_ct_matrix_mul.hpp (29, 93, 112, 147); the 23
     other rotations (Bootstrapper.cpp) use gal_keys_boot.
  2. all four rotate by a multiple of num_batch: i*num_batch, (col_X-i*g)*num_batch, j*num_batch,
     j*g*num_batch.
  3. num_batch = num_X = 256 (single_att_block(..., gal_keys, bootstrapper, num_X, secret_key, 16, layer_id)).
  4. a missing key is decomposed into NAF (signed powers of two, Evaluator::rotate_internal); the NAF of
     256k has only terms +-2^(8+m).
  5. complex_conjugate is called only in Bootstrapper.cpp (boot set, step 0 -> m - 1 via
     gal_steps_vector.push_back(0)).
Reachable set: powers of two from num_X to 2^(logN-2), both signs:
    {+-256, +-512, +-1024, +-2048, +-4096, +-8192, +-16384}   = 14 elements
    default `create_galois_keys(gal_keys)`                     = 1 + 2*(logN-1) = 31 elements
17 of 31 keys (~24.1 GB) generated and never used.

Free (less work: keys never allocated or built).
Donor: retired 02a_minimal_attention_keys (same edit and bound:
`for (int step = num_X; step <= 1 << (logN-2); step <<= 1)`); numbers not cited.
Falsifier: a rotation step not a multiple of num_X throws invalid_argument("Galois key not present")
in rotate_internal (run aborts).
Order: before park_att_keys (paid, 43.9 GB written and read); afterwards gal_keys ~19.8 GB, the paid
step is re-derived on top.

Usage: python3 patch_minimal_att_keys.py <path to the moai source tree>
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

ANCHOR = "    keygen.create_galois_keys(gal_keys);"

PATCH = """    // m1_minimal_att_keys: `gal_keys` is read ONLY by Ct_ct_matrix_mul, and only at multiples
    // of num_batch (= num_X). SEAL's NAF decomposition of 256k has terms +-2^(8+m) only, so no
    // step below num_X can ever be requested and no conjugation is needed on this path. The
    // default set generates 31 elements; 14 are reachable. See impl/patch_minimal_att_keys.py.
    vector<int> att_steps;
    const int snni_max_rot = 1 << (logN - 2);
    for (int snni_step = num_X; snni_step <= snni_max_rot; snni_step <<= 1) {
        att_steps.push_back(snni_step);
        att_steps.push_back(-snni_step);
    }
    keygen.create_galois_keys(att_steps, gal_keys);
    // `logN` is a `long` here, so the default-set size is cast rather than handed to %d as-is.
    // The same class of slip cost a BumbleBee image build an hour earlier today, where
    // -Werror=format caught it; this build has no -Werror and would have printed garbage instead.
    fprintf(stderr, "LEVER|minimal_att_keys|requested=%zu|default_would_be=%d|num_X=%d|max_rot=%d\\n",
            att_steps.size(), (int)(1 + 2 * (logN - 1)), num_X, snni_max_rot);
    fflush(stderr);
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "minimal_att_keys" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the default key generation matched {txt.count(ANCHOR)} times, expected 1")

    # THE BOUND DEPENDS ON `num_X` AND ON `logN`, so both must be in scope AND be what the proof
    # assumes. A tree where `num_X` had been renamed, or where the attention were handed something
    # other than `num_X` as its `num_batch`, would make the bound wrong while still compiling.
    if "const int num_X = " not in txt:
        sys.exit("FAILED: no `num_X` declaration; the lower bound of the reachable set is derived "
                 "from it")
    if "gal_keys, bootstrapper, num_X" not in txt.replace("\n", " ").replace("  ", " "):
        sys.exit("FAILED: the attention is not handed `num_X` as its num_batch in this tree, so "
                 "the proof that every rotation is a multiple of num_X does not hold here. "
                 "Re-derive the bound rather than loosening it.")

    txt = txt.replace(ANCHOR, PATCH, 1)
    open(src, "w").write(txt)
    for what in ("att_steps", "LEVER|minimal_att_keys"):
        if what not in open(src).read():
            sys.exit(f"FAILED: applied the patch but {what!r} is absent")
    print("patch_minimal_att_keys: attention Galois keys reduced to the reachable set (" + SRC + ")")


if __name__ == "__main__":
    main()
