#!/usr/bin/env python3
"""MOAI-CPU x_stream_matmul: attention input X streamed from disk one ciphertext at a time instead of
holding all 768.

    patch_x_stream_matmul.py <moai source tree>

After m20 (key park) and m23 (pool threshold) the matmul window is the peak (~54 GiB): rotation keys
(17.2 GB, live), X (enc_ecd_x, 768 cts, 12.1 GB), Q/K/V and QK^T temporaries. X is read by the Q and K
plaintext matmuls (ct_pt_matrix_mul_wo_pre) and the enc_X_v copy. Written once per layer (one file per
ciphertext); restructured kernel (outer loop over X, inner over the 64 output columns; per-output add
order unchanged). X: 12.1 GB -> one ciphertext. Paid (3 reads of X per head, 24 heads per run). Gate
unchanged.
"""
import os
import sys

ATT = "include/source/att_block/single_att_block.hpp"
TFS = "include/test/test_full_scheme.hpp"

HELPER_ANCHOR = "vector<Ciphertext> single_att_block("
HELPER = """// x_stream_matmul: the attention input lives on disk, one file per ciphertext, for the whole block.
// snni_x_stream_n > 0 means "X is streamed, enc_X is empty". Files in the run directory (MOAI_PARK_DIR
// overrides it); the NAME matters: harvest_step.sh excludes key_*.dat.
static int snni_x_stream_n = 0;
static std::string snni_x_stream_path(int j) {
    const char *d = getenv("MOAI_PARK_DIR");
    std::string dir = (d && *d) ? std::string(d) : std::string(".");
    return dir + "/key_xstream_" + std::to_string(j) + ".dat";
}
static void snni_x_stream_load(Ciphertext &c, const SEALContext &ctx, int j) {
    std::ifstream in(snni_x_stream_path(j), std::ios::binary);
    if (!in) {
        fprintf(stderr, "LEVER|x_stream_matmul|FAILED|cannot open %s\\n", snni_x_stream_path(j).c_str());
        fflush(stderr);
        exit(9);
    }
    c.load(ctx, in);
}
// The plaintext matmul of ct_pt_matrix_mul_wo_pre with the loops swapped: outer over the input
// ciphertexts (streamed), inner over the output columns. Per output column the operations and their
// order are those of the original kernel (X0*w0, then += Xj*wj for j = 1..), so the bytes are the same.
static vector<Ciphertext> snni_ct_pt_matrix_mul_stream(const vector<vector<double>> &W, int num_col,
                                                       int col_W, const SEALContext &seal_context) {
    vector<Ciphertext> output(col_W);
    CKKSEncoder encoder(seal_context);
    Evaluator evaluator(seal_context, encoder);
    double scale = 0;
    for (int j = 0; j < num_col; ++j) {
        Ciphertext xj;
        snni_x_stream_load(xj, seal_context, j);
        if (j == 0) scale = xj.scale();
        #pragma omp parallel for
        for (int i = 0; i < col_W; ++i) {
            Plaintext w;
            encoder.encode(W[j][i], xj.parms_id(), xj.scale(), w);
            if (j == 0) {
                evaluator.multiply_plain(xj, w, output[i]);
            } else {
                Ciphertext temp;
                evaluator.multiply_plain(xj, w, temp);
                evaluator.add_inplace(output[i], temp);
            }
        }
    }
    for (int i = 0; i < col_W; ++i) {
        evaluator.rescale_to_next_inplace(output[i]);
        output[i].scale() = scale;
    }
    return output;
}

vector<Ciphertext> single_att_block("""

SCALE_OLD = "  double scale = enc_X[0].scale();\n"
SCALE_NEW = """  Ciphertext _snni_x0;
  if (snni_x_stream_n) snni_x_stream_load(_snni_x0, seal_context, 0);
  double scale = snni_x_stream_n ? _snni_x0.scale() : enc_X[0].scale();
"""
NCOL_OLD = "  int num_col = enc_X.size();\n"
NCOL_NEW = "  int num_col = snni_x_stream_n ? snni_x_stream_n : (int)enc_X.size();\n"
Q_OLD = "  vector<Ciphertext> Q = ct_pt_matrix_mul_wo_pre(enc_X, WQ, num_col, col_W, num_col, seal_context);\n"
Q_NEW = "  vector<Ciphertext> Q = snni_x_stream_n ? snni_ct_pt_matrix_mul_stream(WQ, num_col, col_W, seal_context)\n                          : ct_pt_matrix_mul_wo_pre(enc_X, WQ, num_col, col_W, num_col, seal_context);\n"
K_OLD = "  vector<Ciphertext> K = ct_pt_matrix_mul_wo_pre(enc_X, WK, num_col, col_W, num_col, seal_context);\n"
K_NEW = "  vector<Ciphertext> K = snni_x_stream_n ? snni_ct_pt_matrix_mul_stream(WK, num_col, col_W, seal_context)\n                          : ct_pt_matrix_mul_wo_pre(enc_X, WK, num_col, col_W, num_col, seal_context);\n"
V_OLD = "      enc_X_v[i] = enc_X[i];\n"
V_NEW = "      if (snni_x_stream_n) snni_x_stream_load(enc_X_v[i], seal_context, i); else enc_X_v[i] = enc_X[i];\n"

LOOP_ANCHOR = """    for (int i = 0; i < 12; ++i){
        att_block[i] = single_att_block(enc_ecd_x, WQ[i], WK[i], WV[i], bQ[i], bK[i], bV[i], b_vec, num_input,
"""
LOOP_NEW = """    // x_stream_matmul: write the block's input to disk, one file per ciphertext, and free it; the
    // heads read it back one ciphertext at a time (snni_ct_pt_matrix_mul_stream, enc_X_v).
    {
        int _snni_nx = (int)enc_ecd_x.size();
        for (int _j = 0; _j < _snni_nx; ++_j) {
            std::ofstream out(snni_x_stream_path(_j), std::ios::binary);
            if (!out) {
                fprintf(stderr, "LEVER|x_stream_matmul|FAILED|cannot open %s\\n", snni_x_stream_path(_j).c_str());
                fflush(stderr);
                exit(9);
            }
            enc_ecd_x[_j].save(out, compr_mode_type::none);
        }
        vector<Ciphertext>().swap(enc_ecd_x);
        malloc_trim(0);
        snni_x_stream_n = _snni_nx;
        fprintf(stderr, "LEVER|x_stream_matmul|layer=%d|action=streamed|n=%d\\n", layer_id, _snni_nx);
        fflush(stderr);
    }
""" + LOOP_ANCHOR
END_ANCHOR = """    gettimeofday(&tend1,NULL);
    double att_block_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
"""
END_NEW = """    // x_stream_matmul: the block is over; drop the stream files (enc_ecd_x stays empty, the release
    // below finds nothing to release and the layer-end loop resizes it as before).
    {
        for (int _j = 0; _j < snni_x_stream_n; ++_j) remove(snni_x_stream_path(_j).c_str());
        fprintf(stderr, "LEVER|x_stream_matmul|layer=%d|action=cleared|n=%d\\n", layer_id, snni_x_stream_n);
        fflush(stderr);
        snni_x_stream_n = 0;
    }
""" + END_ANCHOR


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    a = os.path.join(sys.argv[1], ATT)
    t = os.path.join(sys.argv[1], TFS)
    for p in (a, t):
        if not os.path.exists(p):
            sys.exit("FAILED: no " + p)
    ta = open(a).read()
    tt = open(t).read()
    if "x_stream_matmul" in ta or "x_stream_matmul" in tt:
        sys.exit("FAILED: this tree already carries the lever")
    for name, s, txt in (("helper", HELPER_ANCHOR, ta), ("scale", SCALE_OLD, ta), ("num_col", NCOL_OLD, ta),
                         ("Q", Q_OLD, ta), ("K", K_OLD, ta), ("V", V_OLD, ta),
                         ("head loop", LOOP_ANCHOR, tt), ("loop end", END_ANCHOR, tt)):
        if txt.count(s) != 1:
            sys.exit(f"FAILED: the {name} anchor appears {txt.count(s)} times, expected 1")
    if "#include <fstream>" not in ta:
        sys.exit("FAILED: single_att_block.hpp lacks <fstream> (expected from att_keys_park_softmax)")
    # no other reader of enc_X may remain in the block
    rest = ta.replace(SCALE_OLD, "").replace(NCOL_OLD, "").replace(Q_OLD, "").replace(K_OLD, "").replace(V_OLD, "")
    body = rest[rest.index("vector<Ciphertext> single_att_block("):]
    if "enc_X[" in body or "enc_X." in body:
        sys.exit("FAILED: single_att_block still reads enc_X somewhere else; the stream would miss it")
    ta = (ta.replace(HELPER_ANCHOR, HELPER, 1).replace(SCALE_OLD, SCALE_NEW, 1).replace(NCOL_OLD, NCOL_NEW, 1)
            .replace(Q_OLD, Q_NEW, 1).replace(K_OLD, K_NEW, 1).replace(V_OLD, V_NEW, 1))
    tt = tt.replace(LOOP_ANCHOR, LOOP_NEW, 1).replace(END_ANCHOR, END_NEW, 1)
    open(a, "w").write(ta)
    open(t, "w").write(tt)
    print("patch_x_stream_matmul: X streamed from disk through the attention block (" + ATT + ", " + TFS + ")")


if __name__ == "__main__":
    main()
