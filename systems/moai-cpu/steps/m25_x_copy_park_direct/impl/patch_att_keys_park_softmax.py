#!/usr/bin/env python3
"""MOAI-CPU att_keys_park_softmax: attention rotation keys on disk during each head's softmax bootstrap.

    patch_att_keys_park_softmax.py <moai source tree>

Basis: m20 attention-block peak (layer 1, 68.5 GiB): softmax bootstrap phase keys (19.8 GB, read) and
attention rotation keys gal_keys (17.2 GB, read only by QK^T and softmax*V; the bootstrap uses
bootstrapper_att.gal_keys). gal_keys written once, dropped before every softmax bootstrap, read back
before softmax*V. Paid (one write per run, one 17.2 GB read per head, 24 per run). Same bytes
(compr_mode none, as park_att_keys); gate unchanged.
"""
import os
import sys

SRC = "include/source/att_block/single_att_block.hpp"

INCLUDE_ANCHOR = "#include <chrono>\n#include <malloc.h>\n"
INCLUDE = "#include <chrono>\n#include <malloc.h>\n#include <fstream>\n#include <cstdio>\n#include <cstdlib>\n"

SIG_ANCHOR = "  const SEALContext& seal_context, const RelinKeys &relin_keys, const GaloisKeys & RotK, Bootstrapper &bootstrapper_att,\n"
SIG = "  const SEALContext& seal_context, const RelinKeys &relin_keys, GaloisKeys & RotK, Bootstrapper &bootstrapper_att,\n"

HELPER_ANCHOR = "vector<Ciphertext> single_att_block("
HELPER = """// att_keys_park_softmax: the rotation keys live on disk while a head's softmax bootstrap runs. The
// file is written once per run (the same bytes every layer: park_att_keys reloads them unchanged)
// in the run directory, MOAI_PARK_DIR overrides it. The NAME matters: harvest_step.sh excludes key_*.dat.
static std::string snni_att_softmax_key_path() {
    const char *d = getenv("MOAI_PARK_DIR");
    std::string dir = (d && *d) ? std::string(d) : std::string(".");
    return dir + "/key_gal_softmax_parked.dat";
}
static void snni_att_keys_park_softmax(GaloisKeys &k, int layer_id) {
    static bool _snni_written = false;
    std::string p = snni_att_softmax_key_path();
    if (!_snni_written) {
        std::ofstream out(p, std::ios::binary);
        if (!out) {
            fprintf(stderr, "LEVER|att_keys_park_softmax|FAILED|cannot open %s\\n", p.c_str());
            fflush(stderr);
            exit(9);
        }
        k.save(out, compr_mode_type::none);
        _snni_written = true;
    }
    k = GaloisKeys();
    malloc_trim(0);
    fprintf(stderr, "LEVER|att_keys_park_softmax|layer=%d|action=parked\\n", layer_id);
    fflush(stderr);
}
static void snni_att_keys_unpark_softmax(GaloisKeys &k, const SEALContext &ctx, int layer_id) {
    std::string p = snni_att_softmax_key_path();
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        fprintf(stderr, "LEVER|att_keys_park_softmax|FAILED|cannot reopen %s\\n", p.c_str());
        fflush(stderr);
        exit(9);
    }
    k.load(ctx, in);
    fprintf(stderr, "LEVER|att_keys_park_softmax|layer=%d|action=reloaded\\n", layer_id);
    fflush(stderr);
}

vector<Ciphertext> single_att_block("""

CALL_ANCHOR = """    vector<Ciphertext> enc_softmax = softmax_boot(QK,bias_vec,input_num,seal_context,
        relin_keys,iter,sk,bootstrapper_att, layer_id);
"""
CALL = """    // att_keys_park_softmax: the softmax bootstrap reads bootstrapper_att's keys, never RotK.
    snni_att_keys_park_softmax(RotK, layer_id);
    vector<Ciphertext> enc_softmax = softmax_boot(QK,bias_vec,input_num,seal_context,
        relin_keys,iter,sk,bootstrapper_att, layer_id);
    snni_att_keys_unpark_softmax(RotK, seal_context, layer_id);
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()
    if "att_keys_park_softmax" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, a in (("include", INCLUDE_ANCHOR), ("signature", SIG_ANCHOR), ("helper", HELPER_ANCHOR), ("call", CALL_ANCHOR)):
        if txt.count(a) != 1:
            sys.exit(f"FAILED: the {name} anchor appears {txt.count(a)} times, expected 1")
    body = txt[txt.index(CALL_ANCHOR):]
    if "RotK" not in body:
        sys.exit("FAILED: no RotK reader after the softmax call (nothing to reload for)")
    txt = (txt.replace(INCLUDE_ANCHOR, INCLUDE, 1).replace(SIG_ANCHOR, SIG, 1)
              .replace(HELPER_ANCHOR, HELPER, 1).replace(CALL_ANCHOR, CALL, 1))
    open(src, "w").write(txt)
    print("patch_att_keys_park_softmax: rotation keys parked around every head's softmax bootstrap (" + SRC + ")")


if __name__ == "__main__":
    main()
