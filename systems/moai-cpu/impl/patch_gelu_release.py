#!/usr/bin/env python3
"""MOAI-CPU m1_gelu_release: release each power x_n[i] at the end of the iteration that last reads it (x_n[24] is the mod-switch reference, so release at iteration end, not up front). FREE.

    patch_gelu_release.py <moai source tree>
"""
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
ROOT = sys.argv[1]
SRC = os.path.join(ROOT, "include", "source", "non_linear_func", "gelu_others.hpp")
if not os.path.isfile(SRC):
    sys.exit(f"FAILED: {SRC} is not where this patch expects it")

s = open(SRC, encoding="utf-8").read()

MARK = "m1_gelu_release"
if MARK in s:
    print("  already present: gelu_release")
    sys.exit(0)

# Anchor on the whole if/else block (trailing whitespace included) so the release lands at the end
# of the iteration; x_n[24] is the per-iteration mod-switch reference and must outlive earlier ones.
OLD = "    if (i == 1){\n      res = x_n[i];\n    }\n    else{\n      evaluator.add_inplace(res,x_n[i]);\n    }   \n"
NEW = """    if (i == 1){
      res = x_n[i];
    }
    else{
      evaluator.add_inplace(res,x_n[i]);
    }   

    // m1_gelu_release: this power has now been read for the last time.
    //
    // `x_n` holds all twenty-five until gelu_v2 returns, and at m0's peak that is 10,818 live
    // ciphertexts of 21.7 MB, rank 1 of the object table at 98%. Assigning an empty Ciphertext
    // destroys this one's buffer here instead of at the closing brace of the function.
    //
    // AT THE END OF THE ITERATION, NOT EARLIER: line 129 reads `x_n[24].parms_id()` on every
    // iteration, so x_n[24] must outlive all of them. Releasing x_n[i] in iteration i does that
    // by construction; releasing the vector up front would not.
    x_n[i] = Ciphertext();
    {
      // Announced ONCE, not per call: gelu_v2 runs about 433 times in parallel per FFN and a
      // per-call line would drown the log. The build greps the BINARY for this string, so a
      // rebuild that dropped the patch fails instead of measuring the baseline under this name.
      static bool snni_gelu_release_said = false;
      if (!snni_gelu_release_said) {
        snni_gelu_release_said = true;
        fprintf(stderr, "LEVER|gelu_release|powers=24|released_after=last_read\\n");
        fflush(stderr);
      }
    }
"""
if s.count(OLD) != 1:
    sys.exit(
        f"FAILED: the summation block matched {s.count(OLD)} times in {SRC}, expected exactly 1. "
        f"The release must go at the end of the iteration that read the power, so this patch "
        f"refuses rather than guess a second site."
    )
s = s.replace(OLD, NEW, 1)

# The loop must still read x_n[24] as its mod-switch reference, or the ordering argument fails.
if "mod_switch_to_inplace(x_n[i],x_n[24].parms_id())" not in s.replace(" ", ""):
    sys.exit("FAILED: the x_n[24] mod-switch reference is not where the ordering argument needs it")

open(SRC, "w", encoding="utf-8").write(s)
print("  patched: each power released after its last read (gelu_others.hpp)")
print("  marker  : LEVER|gelu_release")
