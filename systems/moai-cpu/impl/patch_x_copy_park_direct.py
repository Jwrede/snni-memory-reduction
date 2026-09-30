#!/usr/bin/env python3
"""MOAI-CPU x_copy_park_direct: at the layer switch, write the next layer's residual copy to its park
file directly from the bootstrap output (no materialised copy).

    patch_x_copy_park_direct.py <moai source tree>

After bootstrap4 the layer switch copies rtn2 into enc_ecd_x (next input) and enc_ecd_x_copy
(residual), then parks the copy (m17): three 16.9 GB arrays resident; the run's peak once the startup
window is fixed (54.0 GiB in m25x: two 16.9 GB rows at the copy line, rtn2 16.9, context 3.5). save() of
a copy-assigned ciphertext = save() of rtn2. Nothing reads enc_ecd_x_copy before its reload at
residual1 (last layer: never). Same bytes, same reload; gate unchanged. Free.
Requires the park_x_copy tree (anchors).
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

HELPER_ANCHOR = "static void snni_unpark_cts(std::vector<seal::Ciphertext> &v, const seal::SEALContext &ctx, const char *what) {\n"
HELPER = """// x_copy_park_direct: the park file written from the source array, no copy materialised.
static void snni_park_cts_from(const std::vector<seal::Ciphertext> &src, const char *what) {
    std::ofstream out(snni_x_copy_path(), std::ios::binary);
    if (!out) {
        fprintf(stderr, "LEVER|park_x_copy|FAILED|cannot open %s\\n", snni_x_copy_path().c_str());
        fflush(stderr);
        exit(9);
    }
    uint64_t n = src.size();
    out.write(reinterpret_cast<const char *>(&n), sizeof n);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i].save(out, seal::compr_mode_type::none);
    }
    out.close();
    malloc_trim(0);
    fprintf(stderr, "LEVER|x_copy_park_direct|%s|n=%llu\\n", what, (unsigned long long)n);
    fflush(stderr);
}
""" + HELPER_ANCHOR

SWITCH_ANCHOR = """    for(int _k=0;_k<768;++_k){ enc_ecd_x[_k] = rtn2[_k]; enc_ecd_x_copy[_k] = rtn2[_k]; }
    if (layer_id + 1 < snni_n_layers) {
        snni_park_cts(enc_ecd_x_copy, "layer_end");
    }
"""
SWITCH = """    for(int _k=0;_k<768;++_k){ enc_ecd_x[_k] = rtn2[_k]; }
    // x_copy_park_direct: the residual copy for the next layer is written to its park file straight
    // from rtn2 (the bytes a copy would carry) and never materialised; nothing reads it before the
    // reload at residual1, and in the last layer nothing reads it at all.
    vector<Ciphertext>().swap(enc_ecd_x_copy);
    if (layer_id + 1 < snni_n_layers) {
        snni_park_cts_from(rtn2, "layer_end");
    }
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "x_copy_park_direct" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, anchor in (("helper", HELPER_ANCHOR), ("switch", SWITCH_ANCHOR)):
        n = txt.count(anchor)
        if n != 1:
            sys.exit(f"FAILED: the {name} anchor appears {n} times, expected 1")
    # no reader of enc_ecd_x_copy between the switch and the end of the layer loop body
    tail = txt[txt.index(SWITCH_ANCHOR) + len(SWITCH_ANCHOR):]
    body_end = tail.index("\n}\n")  # end of all_layer_test's layer loop / function
    import re
    reads = [m.start() for m in re.finditer(r"\benc_ecd_x_copy\b", tail[:body_end])
             if not tail[max(0, m.start() - 4):m.start()].strip().startswith("//")]
    reads = [p for p in reads if "//" not in tail[tail.rfind("\n", 0, p) + 1:p]]
    if reads:
        sys.exit(f"FAILED: enc_ecd_x_copy is used {len(reads)} time(s) after the layer switch")
    txt = txt.replace(HELPER_ANCHOR, HELPER, 1)
    txt = txt.replace(SWITCH_ANCHOR, SWITCH, 1)
    open(src, "w").write(txt)
    out = open(src).read()
    if out.count('snni_park_cts_from(rtn2, "layer_end")') != 1 or out.count('snni_park_cts(enc_ecd_x_copy, "layer_end")') != 0:
        sys.exit("FAILED: post-check")
    print("OK: x_copy_park_direct applied to " + src)


if __name__ == "__main__":
    main()
