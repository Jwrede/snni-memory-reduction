#!/usr/bin/env python3
"""MOAI-CPU m1_pool_new: SEAL memory manager -> MMProfNew (fresh per-request pool, freed on handle
death) at the top of main. Enabling step.

    patch_pool_new.py <moai-tree>

Image build, after pin_seed_and_gate.py (anchor: main() as that script leaves it).
"""
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/moai")
test_cpp = root / "test.cpp"


def edit(path, anchor, addition, already, what):
    """Insert `addition` after `anchor`, idempotently, with the anchor count checked.
    `already` is passed explicitly, not derived, to avoid matching unpatched source and skipping.
    """
    s = path.read_text()
    if already in s:
        print(f"  already patched: {what}")
        return
    n = s.count(anchor)
    if n != 1:
        sys.exit(f"FAILED: anchor for '{what}' appears {n} times in {path.name}, expected 1. "
                 f"The source moved; re-derive the patch.")
    path.write_text(s.replace(anchor, anchor + addition, 1))
    if already not in path.read_text():
        sys.exit(f"FAILED: applied '{what}' but its marker is absent from {path.name}")
    print(f"  patched: {what}  ({path.name})")


# `include.hpp` already pulls in seal/seal.h (MemoryManager, MMProfNew); only <memory> is added,
# for make_unique, which nothing test.cpp includes guarantees.
edit(
    test_cpp,
    '#include "include.hpp"',
    "\n#include <memory>   // SNNI m1_pool_new: std::make_unique",
    "SNNI m1_pool_new: std::make_unique",
    "the <memory> include",
)

# Before anything allocates: the profile applies at request time, so switching later would leave
# the early-allocated (and largest) key material in the global pool.
edit(
    test_cpp,
    "int main(){",
    """
    // SNNI m1_pool_new: every request gets a fresh pool that dies with its handle, instead of the
    // one global pool that never returns a byte to the OS. The marker is what the image build
    // greps for, so a silently unapplied patch cannot produce a run that looks like this step.
    seal::MemoryManager::SwitchProfile(std::make_unique<seal::MMProfNew>());
    std::cout << "SNNI_MM|profile=new" << std::endl;""",
    "SNNI_MM|profile=new",
    "the MMProfNew switch at the top of main",
)

print("patch_pool_new: done")
