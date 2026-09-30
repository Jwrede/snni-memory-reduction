#!/usr/bin/env python3
"""MOAI-CPU pool_scope_matmul: route the ct_ct rotation phase (rot_enc_X) through a local free-list pool instead of the global threshold pool, so reused rotation key-switch buffers are cached rather than mmap/munmap'd per rotation. Ports the earlier campaign's Exp 14 fix. RUNTIME only; peak unchanged (output is allocated before the scope).

    patch_pool_scope_matmul.py <moai source root>
"""
import os
import sys

SRC = "include/source/matrix_mul/Ct_ct_matrix_mul.hpp"

# Open the local reuse pool right before rot_enc_X is built (the rotation phase), and restore the
# global profile right before the baby-step/giant-step multiply. rot_enc_X keeps the pool alive via
# its ciphertexts' handles until the function returns.
RELEASE_ANCHOR = "  vector<Ciphertext> rot_enc_X(row_X);\n"
RELEASE_PATCH = (
    "  // LEVER pool_scope_matmul: rotation key-switch buffers are reused across rows, so the global\n"
    "  // MemoryPoolThresholdMT would mmap/munmap per rotation (100x, earlier campaign Exp 14). Route\n"
    "  // the rotation phase through a local free-list pool; the global profile is restored before the\n"
    "  // multiply, and output (declared earlier) is untouched so the peak is unchanged.\n"
    "  MemoryPoolHandle _snni_rot_pool = MemoryPoolHandle::New(true);\n"
    "  auto _snni_old_prof = MemoryManager::SwitchProfile(make_unique<MMProfFixed>(_snni_rot_pool));\n"
    '  { static bool _once=false; if(!_once){_once=true; fprintf(stderr,"LEVER|pool_scope_matmul|rot_pool\\n"); fflush(stderr);} }\n'
    "  vector<Ciphertext> rot_enc_X(row_X);\n"
)

RESTORE_ANCHOR = "  //baby step + gaint step (col_w times)\n"
RESTORE_PATCH = (
    "  MemoryManager::SwitchProfile(move(_snni_old_prof));\n"
    "  //baby step + gaint step (col_w times)\n"
)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "pool_scope_matmul" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, anch in (("rot_enc_X release anchor", RELEASE_ANCHOR),
                       ("baby-step restore anchor", RESTORE_ANCHOR)):
        if txt.count(anch) != 1:
            sys.exit(f"FAILED: {name} matched {txt.count(anch)} times, expected 1")
    # release must come before restore.
    if txt.index(RELEASE_ANCHOR) > txt.index(RESTORE_ANCHOR):
        sys.exit("FAILED: rot_enc_X anchor is after the baby-step anchor")

    txt = txt.replace(RELEASE_ANCHOR, RELEASE_PATCH, 1)
    txt = txt.replace(RESTORE_ANCHOR, RESTORE_PATCH, 1)
    open(src, "w").write(txt)
    out = open(src).read()
    for what in ("LEVER|pool_scope_matmul", "SwitchProfile(make_unique<MMProfFixed>",
                 "SwitchProfile(move(_snni_old_prof))"):
        if what not in out:
            sys.exit(f"FAILED: applied the patch but {what!r} is absent")
    if out.index("MMProfFixed>(_snni_rot_pool)") > out.index("SwitchProfile(move(_snni_old_prof))"):
        sys.exit("FAILED: restore ended up before the switch")
    print("patch_pool_scope_matmul: ct_ct rotation phase routed through a local free-list pool (" + SRC + ")")


if __name__ == "__main__":
    main()
