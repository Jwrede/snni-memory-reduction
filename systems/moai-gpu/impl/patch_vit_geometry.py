#!/usr/bin/env python3
"""MOAI-GPU: ViT-Base/16 geometry (sequence length in three places).

    patch_vit_geometry.py <moai source tree>

Sequence length also appears as literals in softmax.cuh:117 (softmax) and softmax.cuh:486
(softmax_boot), slot_count / 128, which cannot read num_row. ViT (197 tokens) padded to 256, batch
halved: num_X 256 -> 128, num_row 128 -> 256 (32,768 slots, same packed volume). Literals replaced by
SNNI_NUM_ROW (default 128, baseline unchanged); SNNI_GEOM| observable, abort on stride mismatch (the
T1s gate is stride-blind). Never a waterfall row.
"""
import os
import sys

TEST = "src/include/test/test_single_layer.cuh"
SOFT = "src/include/source/non_linear_func/softmax.cuh"

MACRO = """// SNNI ViT geometry probe: the sequence length, parameterised. Defaults to the measured BERT
// value, so an image built without -DSNNI_NUM_ROW behaves exactly as upstream does.
#ifndef SNNI_NUM_ROW
#define SNNI_NUM_ROW 128
#endif
#ifndef SNNI_NUM_X
#define SNNI_NUM_X 256
#endif
// A stride mismatch is invisible to this system's STRUCTURAL gate, so it is made fatal here.
static inline void snni_geom_check(size_t slot_count, size_t num_batch, const char *where) {
    static int announced = 0;
    if (!announced) {
        fprintf(stderr, "SNNI_GEOM|slot_count=%zu|num_row=%d|num_batch=%zu|num_X=%d|at=%s\\n",
                slot_count, (int)SNNI_NUM_ROW, num_batch, (int)SNNI_NUM_X, where);
        fflush(stderr);
        announced = 1;
    }
    if (num_batch * (size_t)SNNI_NUM_ROW != slot_count || num_batch != (size_t)SNNI_NUM_X) {
        fprintf(stderr, "SNNI_GEOM|FATAL|%s: num_batch %zu * num_row %d != slot_count %zu, "
                        "or num_batch != num_X %d\\n",
                where, num_batch, (int)SNNI_NUM_ROW, slot_count, (int)SNNI_NUM_X);
        fflush(stderr);
        abort();
    }
}
"""


def edit(path, old, new, what, already, count=1):
    src = open(path).read()
    if already in src:
        print(f"  already patched: {what}")
        return
    n = src.count(old)
    if n != count:
        sys.exit(f"FAILED: anchor for '{what}' appears {n} times, expected {count}. Upstream moved; "
                 f"re-derive the patch rather than loosen it.")
    open(path, "w").write(src.replace(old, new, count))
    if already not in open(path).read():
        sys.exit(f"FAILED: applied '{what}' but its marker is absent")
    print(f"  patched: {what}")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    test, soft = os.path.join(root, TEST), os.path.join(root, SOFT)
    for p in (test, soft):
        if not os.path.exists(p):
            sys.exit("FAILED: no " + p)

    # The macro block goes at the top of softmax.cuh, after its own include, so both call sites see
    # it and so `abort` and `fprintf` are declared.
    edit(soft, '#include "include.cuh"',
         '#include "include.cuh"\n#include <cstdio>\n#include <cstdlib>\n\n' + MACRO,
         "geometry macro and the fatal stride check", already="SNNI_GEOM|")

    edit(soft, "  size_t num_batch = slot_count / 128;",
         "  size_t num_batch = slot_count / SNNI_NUM_ROW;\n"
         "  snni_geom_check(slot_count, num_batch, \"softmax\");",
         "softmax stride", already="snni_geom_check(slot_count, num_batch, \"softmax\")")

    edit(soft, "  int num_batch = slot_count / 128;",
         "  int num_batch = slot_count / SNNI_NUM_ROW;\n"
         "  snni_geom_check(slot_count, (size_t)num_batch, \"softmax_boot\");",
         "softmax_boot stride", already="snni_geom_check(slot_count, (size_t)num_batch")

    # Defaults repeated here on purpose: test_single_layer.cuh uses the macros before include.cuh
    # pulls softmax.cuh, so an upstream reshuffle could otherwise undefine them; #ifndef makes it free.
    edit(test, "#include \"include.cuh\"",
         "#include \"include.cuh\"\n"
         "#ifndef SNNI_NUM_ROW\n#define SNNI_NUM_ROW 128\n#endif\n"
         "#ifndef SNNI_NUM_X\n#define SNNI_NUM_X 256\n#endif",
         "geometry defaults visible where the constants are declared",
         already="#ifndef SNNI_NUM_ROW")

    # The two constants. They are what the geometry IS; the macros above only keep the softmax in
    # step with them.
    edit(test, "const int num_X = 256;", "const int num_X = SNNI_NUM_X;",
         "num_X from the build", already="const int num_X = SNNI_NUM_X;")
    edit(test, "const int num_row = 128;", "const int num_row = SNNI_NUM_ROW;",
         "num_row from the build", already="const int num_row = SNNI_NUM_ROW;")

    print("patch_vit_geometry: geometry parameterised, stride checked, defaults unchanged")


if __name__ == "__main__":
    main()
