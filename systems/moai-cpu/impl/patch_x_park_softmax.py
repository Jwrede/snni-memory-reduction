#!/usr/bin/env python3
"""MOAI-CPU x_park_softmax: attention-block input X on disk during each head's softmax bootstrap.

    patch_x_park_softmax.py <moai source tree>

After att_keys_park_softmax the softmax-bootstrap window holds phase keys (19.8 GB, read) and X
(enc_ecd_x, 768 cts, 12.1 GB; read at the start of each head for Q, K, V only). Written once per layer,
dropped before every softmax bootstrap, read back after it. Paid (one write per layer, one 12.1 GB
read per head). Same bytes (compr_mode none); gate unchanged.
"""
import os
import sys

SRC = "include/source/att_block/single_att_block.hpp"

INCLUDE_ANCHOR = "#include <chrono>\n#include <malloc.h>\n"
INCLUDE = "#include <chrono>\n#include <malloc.h>\n#include <fstream>\n#include <cstdio>\n#include <cstdlib>\n"

SIG_ANCHOR = "vector<Ciphertext> single_att_block(const vector<Ciphertext> & enc_X, \n"
SIG = "vector<Ciphertext> single_att_block(vector<Ciphertext> & enc_X, \n"

HELPER_ANCHOR = "vector<Ciphertext> single_att_block("
HELPER = """// x_park_softmax: the block's input lives on disk while a head's softmax bootstrap runs. The file
// is rewritten at the first head of every layer (X changes per layer) in the run directory,
// MOAI_PARK_DIR overrides it. The NAME matters: harvest_step.sh excludes key_*.dat.
static std::string snni_x_softmax_park_path() {
    const char *d = getenv("MOAI_PARK_DIR");
    std::string dir = (d && *d) ? std::string(d) : std::string(".");
    return dir + "/key_x_softmax_parked.dat";
}
static void snni_x_park_softmax(vector<Ciphertext> &x, int layer_id) {
    static int _snni_written_layer = -1;
    std::string p = snni_x_softmax_park_path();
    if (_snni_written_layer != layer_id) {
        std::ofstream out(p, std::ios::binary);
        if (!out) {
            fprintf(stderr, "LEVER|x_park_softmax|FAILED|cannot open %s\\n", p.c_str());
            fflush(stderr);
            exit(9);
        }
        for (size_t i = 0; i < x.size(); ++i) x[i].save(out, compr_mode_type::none);
        _snni_written_layer = layer_id;
    }
    for (size_t i = 0; i < x.size(); ++i) x[i] = Ciphertext();
    malloc_trim(0);
    fprintf(stderr, "LEVER|x_park_softmax|layer=%d|action=parked|n=%zu\\n", layer_id, x.size());
    fflush(stderr);
}
static void snni_x_unpark_softmax(vector<Ciphertext> &x, const SEALContext &ctx, int layer_id) {
    std::string p = snni_x_softmax_park_path();
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        fprintf(stderr, "LEVER|x_park_softmax|FAILED|cannot reopen %s\\n", p.c_str());
        fflush(stderr);
        exit(9);
    }
    for (size_t i = 0; i < x.size(); ++i) x[i].load(ctx, in);
    fprintf(stderr, "LEVER|x_park_softmax|layer=%d|action=reloaded|n=%zu\\n", layer_id, x.size());
    fflush(stderr);
}

vector<Ciphertext> single_att_block("""

CALL_ANCHOR = """    snni_att_keys_park_softmax(RotK, layer_id);
    vector<Ciphertext> enc_softmax = softmax_boot(QK,bias_vec,input_num,seal_context,
        relin_keys,iter,sk,bootstrapper_att, layer_id);
    snni_att_keys_unpark_softmax(RotK, seal_context, layer_id);
"""
CALL = """    snni_att_keys_park_softmax(RotK, layer_id);
    // x_park_softmax: Q, K and V are computed; X is not read again until the next head.
    snni_x_park_softmax(enc_X, layer_id);
    vector<Ciphertext> enc_softmax = softmax_boot(QK,bias_vec,input_num,seal_context,
        relin_keys,iter,sk,bootstrapper_att, layer_id);
    snni_x_unpark_softmax(enc_X, seal_context, layer_id);
    snni_att_keys_unpark_softmax(RotK, seal_context, layer_id);
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "x_park_softmax" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    have_inc = INCLUDE_ANCHOR in txt or "#include <fstream>" in txt
    for name, a in (("signature", SIG_ANCHOR), ("helper", HELPER_ANCHOR), ("call", CALL_ANCHOR)):
        if txt.count(a) != 1:
            sys.exit(f"FAILED: the {name} anchor appears {txt.count(a)} times, expected 1")
    after = txt[txt.index(CALL_ANCHOR) + len(CALL_ANCHOR):]
    body_end = after.find("\n}\n")
    if "enc_X[" in after[:body_end] or "enc_X)" in after[:body_end]:
        sys.exit("FAILED: enc_X is read after the softmax call inside the block; the park window is wrong")
    if not have_inc:
        sys.exit("FAILED: expected the includes of att_keys_park_softmax to be present")
    txt = txt.replace(SIG_ANCHOR, SIG, 1).replace(HELPER_ANCHOR, HELPER, 1).replace(CALL_ANCHOR, CALL, 1)
    open(src, "w").write(txt)
    print("patch_x_park_softmax: input X parked around every head's softmax bootstrap (" + SRC + ")")


if __name__ == "__main__":
    main()
