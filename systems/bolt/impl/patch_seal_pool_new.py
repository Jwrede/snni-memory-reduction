#!/usr/bin/env python3
"""BOLT p_seal_pool_new (PROBE): switch SEAL's memory manager to MMProfNew at the top of main.
Attacks the resident MemoryPoolHeadMT free list (W=126.8); a probe because two precedents disagree.
    patch_seal_pool_new.py <EzPC tree>
"""
import os
import re
import sys

SRC = "SCI/tests/bert_bolt/bolt_bert.cpp"

ANCHOR = "int main(int argc, char **argv) {\n"

PATCH = """int main(int argc, char **argv) {
    // p_seal_pool_new (PROBE): SEAL's default MMProfGlobal hands every request the ONE global pool
    // and that pool never returns a byte to the OS, so the free list is resident for the whole run.
    // MMProfNew returns a FRESH pool per request, which dies with its handle. Borrowed from
    // MOAI-CPU m1_pool_new; measured here because two precedents disagree and neither is evidence
    // for this line.
    seal::MemoryManager::SwitchProfile(std::make_unique<seal::MMProfNew>());
    fprintf(stderr, "LEVER|seal_pool_new|armed\\n");
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "seal_pool_new" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: main matched {txt.count(ANCHOR)} times, expected exactly 1")

    # SEAL must already be in scope (via he.h/bert.h); adding our own include would make the
    # one-line probe more than one line.
    if not re.search(r'#include\s+"(bert|he)\.h"', txt):
        sys.exit("FAILED: this file includes neither bert.h nor he.h, so seal/seal.h may not be "
                 "in scope; adding an include would make this more than a one-line change")

    txt = txt.replace(ANCHOR, PATCH, 1)
    open(src, "w").write(txt)
    print("patch_seal_pool_new: MMProfNew switched at the top of main (" + SRC + ")")


if __name__ == "__main__":
    main()
