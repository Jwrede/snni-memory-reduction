#!/usr/bin/env python3
"""MOAI-CPU m1_gelu_output_release: release the FFN gelu_output vector (3072 CTs, rank 1 of m0's live table at 57%) at its last read via the file's own vector<Ciphertext>().swap idiom. FREE.

    patch_gelu_output_release.py <moai source tree>
"""
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
ROOT = sys.argv[1]
SRC = os.path.join(ROOT, "include", "test", "test_full_scheme.hpp")
if not os.path.isfile(SRC):
    sys.exit(f"FAILED: {SRC} is not where this patch expects it")

s = open(SRC, encoding="utf-8").read()

MARK = "m1_gelu_output_release"
if MARK in s:
    print("  already present: gelu_output_release")
    sys.exit(0)

# Anchor on the last read (the final matmul call), not the declaration: gelu_output must survive
# until this call returns.
OLD = ("    vector<Ciphertext> final_output = ct_pt_matrix_mul_wo_pre_w_mask(gelu_output, "
       "final_weight,b_vec, num_inter, num_col, num_inter, context);\n")
NEW = OLD + """
    // m1_gelu_output_release: gelu_output has now been read for the last time.
    //
    // 3,072 ciphertexts, filled at line 976 and consumed by the call above. Without this line the
    // vector stays in scope through after_final, after_bootstrap3, after_layernorm2 and
    // after_bootstrap4 -- and the peak is at the last of those. It is rank 1 of m0's live object
    // table at 234,877.6 MB, 57.0% of the peak.
    //
    // The idiom is this file's own: inter_output is released at line 988, final_output after its
    // own last use, and four more besides. gelu_output was the one that was missed.
    vector<Ciphertext>().swap(gelu_output);
    {
        // Announced once. The build greps the BINARY for this string, so a rebuild that dropped
        // the patch fails instead of measuring the baseline under this name.
        static bool snni_gelu_output_said = false;
        if (!snni_gelu_output_said) {
            snni_gelu_output_said = true;
            fprintf(stderr, "LEVER|gelu_output_release|ciphertexts=%d|released_after=last_read\\n",
                    num_inter);
            fflush(stderr);
        }
    }
"""
if s.count(OLD) != 1:
    sys.exit(
        f"FAILED: the final matmul call matched {s.count(OLD)} times in {SRC}, expected exactly 1. "
        f"The release must go after gelu_output's LAST READ, so this patch refuses rather than "
        f"guess the site."
    )

# `gelu_output` must not be read after the anchor (else not retained-past-last-use). Checked on
# the ORIGINAL text, before insertion: the inserted comment names the vector too.
tail = s[s.index(OLD) + len(OLD):]
if "gelu_output" in tail:
    sys.exit(
        "FAILED: `gelu_output` is read after the release point, so it is not retained-past-last-use "
        "as this patch claims. Re-derive the last use before applying."
    )

s = s.replace(OLD, NEW, 1)

open(SRC, "w", encoding="utf-8").write(s)
print("  patched: gelu_output released at its last use (test_full_scheme.hpp)")
print("  marker  : LEVER|gelu_output_release")
