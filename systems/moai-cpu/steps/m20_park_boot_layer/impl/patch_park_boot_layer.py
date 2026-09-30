#!/usr/bin/env python3
"""MOAI-CPU park_boot_layer: boot_layer (copy of bootstrap2's output for the second residual add) on
disk from its creation to that add, instead of resident through FFN, bootstrap3 and layernorm2.

    patch_park_boot_layer.py <moai source tree>

Basis: free-line endpoint m19g (69.6 GiB, layer 1 bootstrap3): phase keys (19.8 GB, live), boot_layer
(16.9 GB, boot_layer[_k] = rtn[_k] after bootstrap2, read only at the residual add before layernorm2),
bootstrap temporaries (~18 GB), pool aggregate. Fix = disk round trip (write 16.9 GB, read back): paid.
Bit-identical (SEAL serialisation, compr_mode none).
Where: park after the `Modulus chain index after bootstrapping` line that reads boot_layer[0]; reload
before the `//rtn+enc_ecd_x_copy` residual loop. The script checks no reference in between. Helpers
as park_x_copy's (file key_boot_layer_parked.dat).
"""
import os
import sys

SRC = "include/test/test_full_scheme.hpp"
HELPER_ANCHOR = "void all_layer_test(){"
HELPER = """// ---------------------------------------------------------------------------------------
// park_boot_layer: bootstrap2's output copy lives on disk from the bootstrap2 log line to the
// second residual add. Same directory rule as park_x_copy (cwd, MOAI_PARK_DIR overrides).
static std::string snni_boot_layer_path() {
    const char *d = getenv("MOAI_PARK_DIR");
    std::string dir = (d && *d) ? std::string(d) : std::string(".");
    return dir + "/key_boot_layer_parked.dat";
}
static void snni_park_boot_layer(std::vector<seal::Ciphertext> &v, const char *what) {
    std::ofstream out(snni_boot_layer_path(), std::ios::binary);
    if (!out) {
        fprintf(stderr, "LEVER|park_boot_layer|FAILED|cannot open %s\\n", snni_boot_layer_path().c_str());
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
    fprintf(stderr, "LEVER|park_boot_layer|%s|action=parked|n=%llu\\n", what, (unsigned long long)n);
    fflush(stderr);
}
static void snni_unpark_boot_layer(std::vector<seal::Ciphertext> &v, const seal::SEALContext &ctx, const char *what) {
    std::ifstream in(snni_boot_layer_path(), std::ios::binary);
    if (!in) {
        fprintf(stderr, "LEVER|park_boot_layer|FAILED|cannot reopen %s\\n", snni_boot_layer_path().c_str());
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
    remove(snni_boot_layer_path().c_str());
    fprintf(stderr, "LEVER|park_boot_layer|%s|action=reloaded|n=%llu\\n", what, (unsigned long long)n);
    fflush(stderr);
}
// ---------------------------------------------------------------------------------------

void all_layer_test(){"""

PARK_ANCHOR = '    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(boot_layer[0].parms_id())->chain_index()<<endl;\n'
PARK = PARK_ANCHOR + '    snni_park_boot_layer(boot_layer, "after_bootstrap2");\n'
UNPARK_ANCHOR = ("    //rtn+enc_ecd_x_copy\n"
                 "    #pragma omp parallel for\n"
                 "\n"
                 "    for (int i = 0; i < num_col; ++i){\n"
                 "        evaluator.mod_switch_to_inplace(boot_layer[i], rtn2[i].parms_id());\n")
UNPARK = '    snni_unpark_boot_layer(boot_layer, context, "residual2");\n' + UNPARK_ANCHOR


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "park_boot_layer" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, anchor in (("helper", HELPER_ANCHOR), ("park", PARK_ANCHOR), ("unpark", UNPARK_ANCHOR)):
        if txt.count(anchor) != 1:
            sys.exit(f"FAILED: {name} anchor appears {txt.count(anchor)} times, expected 1")
    a = txt.index(PARK_ANCHOR) + len(PARK_ANCHOR)
    b = txt.index(UNPARK_ANCHOR)
    if a >= b:
        sys.exit("FAILED: the park point is not before the residual loop")
    reads = [ln for ln in txt[a:b].splitlines() if "boot_layer" in ln and not ln.strip().startswith("//")]
    if reads:
        sys.exit("FAILED: boot_layer is referenced between park and reload:\n" + "\n".join(reads))
    txt = txt.replace(HELPER_ANCHOR, HELPER, 1)
    txt = txt.replace(PARK_ANCHOR, PARK, 1)
    txt = txt.replace(UNPARK_ANCHOR, UNPARK, 1)
    open(src, "w").write(txt)
    out = open(src).read()
    if out.count("snni_park_boot_layer(boot_layer") != 1 or out.count("snni_unpark_boot_layer(boot_layer") != 1:
        sys.exit("FAILED: post-check: park/unpark calls not exactly once each")
    print("patch_park_boot_layer: boot_layer parked after bootstrap2, reloaded before the second residual add (" + SRC + ")")


if __name__ == "__main__":
    main()
