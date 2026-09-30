#!/usr/bin/env python3
"""MOAI-CPU m14_park_x_copy: residual input copy on disk during the attention block, read back before
its next read. Paid (I/O).

    patch_park_x_copy.py <moai source tree>

enc_ecd_x_copy (768 cts, 22 MB each after m9 = 17 GB) is created before the attention block and first
read after bootstrap1 (rtn += enc_ecd_x_copy); dead through the attention block, where the peak sits
once the FFN is tiled and dead keys released. Bytes unchanged; gate unaffected.
Where: park (a) after the layer-0 copy is shrunk (m9's snni_shrink_cts(enc_ecd_x_copy, ...)), (b) after
the re-creation from rtn2 at the end of a layer when another layer follows; unpark before the
rtn+enc_ecd_x_copy residual loop. Requires the m9 tree (anchor (a)).
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"

HELPER_ANCHOR = "void all_layer_test(){"
HELPER = """// ---------------------------------------------------------------------------------------
// m14_park_x_copy: the residual copy lives on disk while the attention block runs. The run
// directory is the process cwd (the runner cds into it); MOAI_PARK_DIR overrides it.
static std::string snni_x_copy_path() {
    const char *d = getenv("MOAI_PARK_DIR");
    std::string dir = (d && *d) ? std::string(d) : std::string(".");
    return dir + "/key_x_copy_parked.dat";
}
static void snni_park_cts(std::vector<seal::Ciphertext> &v, const char *what) {
    std::ofstream out(snni_x_copy_path(), std::ios::binary);
    if (!out) {
        fprintf(stderr, "LEVER|park_x_copy|FAILED|cannot open %s\\n", snni_x_copy_path().c_str());
        fflush(stderr);
        exit(9);
    }
    uint64_t n = v.size();
    out.write(reinterpret_cast<const char *>(&n), sizeof n);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i].save(out, seal::compr_mode_type::none);
    }
    out.close();
    std::vector<seal::Ciphertext>().swap(v);
    malloc_trim(0);
    fprintf(stderr, "LEVER|park_x_copy|%s|action=parked|n=%llu\\n", what, (unsigned long long)n);
    fflush(stderr);
}
static void snni_unpark_cts(std::vector<seal::Ciphertext> &v, const seal::SEALContext &ctx, const char *what) {
    std::ifstream in(snni_x_copy_path(), std::ios::binary);
    if (!in) {
        fprintf(stderr, "LEVER|park_x_copy|FAILED|cannot reopen %s\\n", snni_x_copy_path().c_str());
        fflush(stderr);
        exit(9);
    }
    uint64_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof n);
    v.resize(n);
    for (size_t i = 0; i < n; ++i) {
        v[i].load(ctx, in);
    }
    in.close();
    remove(snni_x_copy_path().c_str());
    fprintf(stderr, "LEVER|park_x_copy|%s|action=reloaded|n=%llu\\n", what, (unsigned long long)n);
    fflush(stderr);
}
// ---------------------------------------------------------------------------------------

void all_layer_test(){"""

# (a) layer 0: right after m9's shrink of the copy.
PARK0_ANCHOR = '    snni_shrink_cts(enc_ecd_x_copy, "enc_ecd_x_copy");\n'
PARK0 = PARK0_ANCHOR + '    snni_park_cts(enc_ecd_x_copy, "layer0");\n'

# (b) later layers: after the re-creation from rtn2, only when another layer follows.
PARKN_ANCHOR = "    for(int _k=0;_k<768;++_k){ enc_ecd_x[_k] = rtn2[_k]; enc_ecd_x_copy[_k] = rtn2[_k]; }\n"
PARKN = PARKN_ANCHOR + """    if (layer_id + 1 < snni_n_layers) {
        snni_park_cts(enc_ecd_x_copy, "layer_end");
    }
"""

# unpark: right before the residual loop that reads the copy.
UNPARK_ANCHOR = """    //rtn+enc_ecd_x_copy
    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i){
        evaluator.mod_switch_to_inplace(enc_ecd_x_copy[i], rtn[i].parms_id());
"""
UNPARK = """    //rtn+enc_ecd_x_copy
    snni_unpark_cts(enc_ecd_x_copy, context, "residual1");
    #pragma omp parallel for

    for (int i = 0; i < num_col; ++i){
        evaluator.mod_switch_to_inplace(enc_ecd_x_copy[i], rtn[i].parms_id());
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "park_x_copy" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, anchor in (("helper", HELPER_ANCHOR), ("park layer 0", PARK0_ANCHOR),
                         ("park layer end", PARKN_ANCHOR), ("unpark", UNPARK_ANCHOR)):
        n = txt.count(anchor)
        if n != 1:
            sys.exit(f"FAILED: the {name} anchor appears {n} times, expected 1")
    if "int snni_n_layers = snni_int_env(" not in txt:
        sys.exit("FAILED: no snni_n_layers in the tree")
    txt = txt.replace(HELPER_ANCHOR, HELPER, 1)
    txt = txt.replace(PARK0_ANCHOR, PARK0, 1)
    txt = txt.replace(PARKN_ANCHOR, PARKN, 1)
    txt = txt.replace(UNPARK_ANCHOR, UNPARK, 1)
    open(src, "w").write(txt)
    out = open(src).read()
    if out.count("snni_park_cts(enc_ecd_x_copy") != 2 or out.count("snni_unpark_cts(enc_ecd_x_copy") != 1:
        sys.exit("FAILED: applied but park/unpark calls are not 2/1")
    print("patch_park_x_copy: enc_ecd_x_copy parked through the attention block (" + SRC + ")")


if __name__ == "__main__":
    main()
