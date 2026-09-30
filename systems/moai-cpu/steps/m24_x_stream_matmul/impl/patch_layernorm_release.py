#!/usr/bin/env python3
"""MOAI-CPU free fix: release dead intermediates inside layernorm/layernorm2.

Port of the gate-safe frees from operator-bench/moai_new/02_phase_keys/layernorm.hpp (decrypted
per-layer output bit-identical, md5 == m0):

  (a) output[i] = nx[i]  ->  output[i] = std::move(nx[i])
      last read of nx[i]; nx and output (768 CTs each) never both full (27.9 GiB layernorm2 object,
      layernorm.hpp:516, m6 table). Same value as a copy.
  (b) vector<Ciphertext>().swap(temp_var) after temp_var is summed into var.
  (c) var = Ciphertext() after invert_sqrt(var,...) consumed it.

Not ported (not frees): num_ct and chunk generalisation (identities at num_ct=768); debug
decrypt/cout blocks kept. layernorm.hpp only (SEAL untouched); layernorm and layernorm2 get the same
three frees.

    patch_layernorm_release.py <moai source root>
"""
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
LN = os.path.join(sys.argv[1], "include", "source", "non_linear_func", "layernorm.hpp")
if not os.path.isfile(LN):
    sys.exit(f"FAILED: {LN} not found")
src = open(LN, encoding="utf-8").read()
if "LEVER|layernorm_release" in src:
    sys.exit("FAILED: this tree already carries the layernorm_release lever")


def edit(old, new, what, count):
    global src
    if src.count(old) != count:
        sys.exit(f"FAILED: '{what}' expected {count} match(es), found {src.count(old)}")
    src = src.replace(old, new, count)


# 0. headers for std::move (utility) and the lever marker fprintf (cstdio)
edit(
    "using namespace std;\nusing namespace seal;\n",
    "#include <cstdio>\n#include <utility>\nusing namespace std;\nusing namespace seal;\n",
    "cstdio/utility includes",
    1,
)

# a. move nx[i] into output[i] instead of copying (layernorm + layernorm2) -- the 27.9 GiB fix
edit(
    "    output[i] = nx[i];\n",
    "    output[i] = std::move(nx[i]);\n",
    "move nx into output",
    2,
)

# b. free temp_var once it has been reduced into var
edit(
    "  evaluator.relinearize_inplace(var,relin_keys);\n"
    "  evaluator.rescale_to_next_inplace(var);\n",
    "  evaluator.relinearize_inplace(var,relin_keys);\n"
    "  evaluator.rescale_to_next_inplace(var);\n"
    "  vector<Ciphertext>().swap(temp_var);\n",
    "free temp_var",
    2,
)

# c. free var after invert_sqrt consumed it, and emit the lever marker once per function
edit(
    "  Ciphertext inv_sqrt_var = invert_sqrt(var,4,2,seal_context,relin_keys);\n",
    "  Ciphertext inv_sqrt_var = invert_sqrt(var,4,2,seal_context,relin_keys);\n"
    "  var = Ciphertext();\n"
    "  { static bool _snni_ln = false; if(!_snni_ln){ _snni_ln = true;"
    " fprintf(stderr, \"LEVER|layernorm_release|active\\n\"); fflush(stderr); } }\n",
    "free var + lever marker",
    2,
)

open(LN, "w", encoding="utf-8").write(src)

out = open(LN, encoding="utf-8").read()
for m in (
    "output[i] = std::move(nx[i]);",
    "vector<Ciphertext>().swap(temp_var);",
    "var = Ciphertext();",
    "LEVER|layernorm_release",
):
    if m not in out:
        sys.exit(f"FAILED: applied but marker {m!r} is absent")
if out.count("std::move(nx[i])") != 2:
    sys.exit(f"FAILED: expected 2 move(nx) edits, found {out.count('std::move(nx[i])')}")
print("patch_layernorm_release: move(nx)+temp_var/var free+marker applied "
      "(include/source/non_linear_func/layernorm.hpp)")
