#!/usr/bin/env python3
"""Paid lever: fused tiled FFN (linear1 -> gelu -> linear2-accumulate per tile).

Target: FFN block at "before intermediate linear": full num_inter (=3072) intermediate inter_output
(~27 GiB, OBJ1) beside rtn (~29 GiB, OBJ0, linear1's input). The 3072 dimension is processed in tiles
of FFN_TILE_SIZE columns: linear1 -> +bias -> gelu, then accumulation into all num_col (768)
final_output columns (linear2 as a running sum over the 3072 rows). Resident: one tile + 768
accumulators.
Bit-identical: accumulation order unchanged (rows 0..3071), same operations; gate md5 == m0. No disk
I/O, ~same compute. Paid-phase lever (structural FFN rewrite), effectively free at runtime; new
relative to s1 (17-lineage does not tile the FFN).

    patch_tiled_ffn.py <moai source root>

Anchors: line range between two unique marker lines of the real tree (works on moai_s1full's
test_full_scheme.hpp after the path-only seds).
"""
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
ROOT = sys.argv[1]
TFS = os.path.join(ROOT, "include", "test", "test_full_scheme.hpp")
if not os.path.isfile(TFS):
    sys.exit(f"FAILED: {TFS} not found")

src = open(TFS, encoding="utf-8").read()
if "SNNI: fused tiled FFN" in src:
    sys.exit("FAILED: this tree already carries the fused tiled FFN lever")

# 1. FFN_TILE_SIZE define, right after the num_inter constant.
NUM_INTER = "const int num_inter = 3072;\n"
if src.count(NUM_INTER) != 1:
    sys.exit(f"FAILED: expected exactly 1 'const int num_inter = 3072;', found {src.count(NUM_INTER)}")
src = src.replace(
    NUM_INTER,
    NUM_INTER
    + "#ifndef FFN_TILE_SIZE\n#define FFN_TILE_SIZE 512\n#endif\n",
    1,
)

# 2. Replace the whole FFN block [linear1 call .. gelu_output swap] with the fused tiled loop.
#    Line-range replacement between two unique marker lines (robust to the commented debug
#    blocks in between).
# PALMA phase-split tree: linear1 uses the _large variant; the FFN block has no swap(gelu_output).
# The fused loop replaces [linear1 .. linear2] (produces final_output pre-bias); the ORIGINAL
# final_bias loop, final_output_size, final_time and after_final marker all survive after it.
START_KEY = "ct_pt_matrix_mul_wo_pre_large(rtn, inter_weight"
END_KEY = "vector<Ciphertext> final_output = ct_pt_matrix_mul_wo_pre_w_mask(gelu_output, final_weight,b_vec, num_inter, num_col, num_inter, context);"

lines = src.split("\n")
starts = [i for i, l in enumerate(lines) if START_KEY in l]
ends = [i for i, l in enumerate(lines) if END_KEY in l]
if len(starts) != 1:
    sys.exit(f"FAILED: START_KEY {START_KEY!r} matched {len(starts)} lines, expected 1")
if len(ends) != 1:
    sys.exit(f"FAILED: END_KEY {END_KEY!r} matched {len(ends)} lines, expected 1")
i_start, i_end = starts[0], ends[0]
if not (i_start < i_end):
    sys.exit("FAILED: END_KEY is not after START_KEY")

FUSED = r"""    // ===== SNNI: fused tiled FFN (linear1 -> gelu -> linear2-accumulate per tile) =====
    // The wide num_inter (3072) intermediate is never fully resident: it is processed in tiles of
    // FFN_TILE_SIZE columns. Each tile runs linear1 (rtn * inter_weight[:,col]) + inter_bias +
    // gelu, then accumulates into all num_col final_output columns (linear2 as a running sum over
    // the 3072 rows). rtn stays resident across tiles (linear1's input), freed after. Accumulation
    // order == the original (rows 0..3071 in order) -> bit-identical, gate == m0. No disk I/O.
    cout <<"Modulus chain index before intermediate linear (fused tiled FFN): "<< context.get_context_data(rtn[0].parms_id())->chain_index()<<endl;
    double inter_time = 0, gelu_time = 0;
    gettimeofday(&tstart1,NULL);
    size_t _snni_slots = encoder.slot_count();
    vector<Ciphertext> final_output(num_col);
    bool _snni_ffn_first = true;
    for (int _tile = 0; _tile < num_inter; _tile += FFN_TILE_SIZE) {
        int _te = min(_tile + FFN_TILE_SIZE, num_inter);
        int _tn = _te - _tile;
        vector<Ciphertext> inter_tile(_tn);
        #pragma omp parallel for
        for (int t = 0; t < _tn; ++t) {
            int col = _tile + t;
            Plaintext _w0;
            encoder.encode(inter_weight[0][col], rtn[0].parms_id(), rtn[0].scale(), _w0);
            evaluator.multiply_plain(rtn[0], _w0, inter_tile[t]);
            for (int j = 1; j < num_col; ++j) {
                Plaintext _wj;
                encoder.encode(inter_weight[j][col], rtn[j].parms_id(), rtn[j].scale(), _wj);
                Ciphertext _tmp;
                evaluator.multiply_plain(rtn[j], _wj, _tmp);
                evaluator.add_inplace(inter_tile[t], _tmp);
            }
            evaluator.rescale_to_next_inplace(inter_tile[t]);
            inter_tile[t].scale() = scale;
            Plaintext _eb;
            vector<double> _bv(_snni_slots, 0);
            for (size_t s = 0; s < _snni_slots; ++s) if (b_vec[s] == 1) _bv[s] = inter_bias[col];
            encoder.encode(_bv, inter_tile[t].parms_id(), inter_tile[t].scale(), _eb);
            evaluator.mod_switch_to_inplace(_eb, inter_tile[t].parms_id());
            inter_tile[t].scale() = scale;
            _eb.scale() = scale;
            evaluator.add_plain_inplace(inter_tile[t], _eb);
            inter_tile[t] = gelu_v2(inter_tile[t], context, relin_keys, secret_key);
        }
        #pragma omp parallel for
        for (int k = 0; k < num_col; ++k) {
            for (int t = 0; t < _tn; ++t) {
                int row = _tile + t;
                Plaintext _fw;
                vector<double> _fwv(_snni_slots, 0);
                for (size_t s = 0; s < _snni_slots; ++s) if (b_vec[s] == 1) _fwv[s] = final_weight[row][k];
                encoder.encode(_fwv, inter_tile[t].parms_id(), inter_tile[t].scale(), _fw);
                Ciphertext _prod;
                evaluator.multiply_plain(inter_tile[t], _fw, _prod);
                if (_snni_ffn_first && t == 0) final_output[k] = _prod;
                else evaluator.add_inplace(final_output[k], _prod);
            }
        }
        _snni_ffn_first = false;
        vector<Ciphertext>().swap(inter_tile);
        jemalloc_purge_all();
        cleanup_main_log_rss(string("ffn_tile_") + to_string(_tile / FFN_TILE_SIZE) + "_done");
    }
    vector<Ciphertext>().swap(rtn);
    #pragma omp parallel for
    for (int k = 0; k < num_col; ++k) {
        evaluator.rescale_to_next_inplace(final_output[k]);
        final_output[k].scale() = scale;
    }
    gettimeofday(&tend1,NULL);
    inter_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"Fused tiled FFN time = "<<inter_time<<endl;
    cleanup_main_log_rss("after_fused_ffn");
    // ===== end SNNI fused tiled FFN ====="""

new_lines = lines[:i_start] + FUSED.split("\n") + lines[i_end + 1:]
out = "\n".join(new_lines)

# 3. Validate.
for token in ("SNNI: fused tiled FFN", "FFN_TILE_SIZE", "end SNNI fused tiled FFN",
              "final_output", "gelu_v2(inter_tile"):
    if token not in out:
        sys.exit(f"FAILED: applied patch but {token!r} absent")
# the original inter_output / gelu_output / linear2 call must be gone from the FFN region
if "ct_pt_matrix_mul_wo_pre_large(rtn, inter_weight" in out:
    sys.exit("FAILED: original linear1 call still present")
if "ct_pt_matrix_mul_wo_pre_w_mask(gelu_output" in out:
    sys.exit("FAILED: original linear2 call still present")
if out.count("{") != out.count("}"):
    sys.exit(f"FAILED: brace imbalance after patch ({out.count('{')} open vs {out.count('}')} close)")
# inter_time/gelu_time/final_time must exist for the later total_time sum
if "double inter_time = 0, gelu_time = 0;" not in out:
    sys.exit("FAILED: inter_time/gelu_time/final_time declaration missing (breaks total_time)")

open(TFS, "w", encoding="utf-8").write(out)
print("patch_tiled_ffn: fused tiled FFN applied (include/test/test_full_scheme.hpp)")
