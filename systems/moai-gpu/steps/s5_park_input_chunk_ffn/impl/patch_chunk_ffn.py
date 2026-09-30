#!/usr/bin/env python3
"""MOAI-GPU s4: FFN output never exists as a whole device array. Paid.

    patch_chunk_ffn.py <moai source tree>

Basis: s3 growth events and object table:
    phase                 used       reserved
    -> after_intermediate +27,648     +3,552    <- the FFN output is built here
    -> after_gelu          +6,144     +8,416    <- the LAST growth event
    -> after_final / bootstrap3 / layernorm2 / bootstrap4:  0

    42,316.3 MB  encrypt_zero_symmetric        1121 allocs   keys, permanent by construction
    29,475.5 MB  mod_switch_scale_to_next      3207 allocs   <- 9.19 MB each: THIS IS inter_output
    16,911.4 MB  boot_layer                     768 allocs   already released by s3
     8,287.9 MB  mod_switch_to_next             826 allocs
     6,427.8 MB  gelu_output                   3065 allocs
3,072 x 9 MiB = 27,648 MiB = after_intermediate growth; largest non-key object.
Not the retired w3 (stream boot4) or w6 (park enc_X): both sites reserve nothing in this tree (fixed
by s2). Nulls acted at before_attention (refilled); s2 (-9.0%), s3 (-10.4%) act at or after the last
growth event; this acts at after_gelu.
Mechanism: retired w5 (s14 in its numbering): matmul in 4 column slices of 768, bias per slice, each
finished slice streamed to the host. Data-bound confirmed here: off-path-runs/s4_ffn_threads/
(8 -> 2 workers: peak +2.95%, wall unchanged).
Paid: slices to the disk bank, read back one ciphertext at a time in the GELU (s5_park_input_copy:
16.9 GB traffic for +0.30% wall; here 27,648 MiB).
Expected: device holds one slice (6,912 MiB) during the matmul, none during the GELU; ~-10,000 MiB
(~-9% of 108,384 MiB; spread 2.94%).
Falsifier: gate T1s (same operands and order; per-slice bias = same addition).

Usage: python3 patch_chunk_ffn.py <path to the moai source tree>
"""
import os
import re
import sys

SRC = "src/include/test/test_single_layer.cuh"

# 1. THE MATMUL. Replaced by the sliced version, which also adds the bias and parks each slice.
MM_ANCHOR = """    vector<PhantomCiphertext> inter_output = ct_pt_matrix_mul_wo_pre_large(rtn, inter_weight, num_col, num_inter, num_col, context);
"""

MM_PATCH = """    // s4_chunk_ffn: the stock code materialises all 3,072 FFN outputs on the device while the key
    // bank and `rtn` are still resident -- 27,648 MiB by the level arithmetic, and the largest
    // non-key object in s3's table. Compute four column slices of 768, add each slice's bias
    // immediately, and park the finished slice off the device. The device then holds the keys,
    // `rtn` and ONE slice. Same operands, same order, same additions; only the lifetime changes.
    const int s4_chunk = 768;
    const int s4_nchunks = num_inter / s4_chunk;
    std::vector<HostCipher> inter_output_host(num_inter);
    for (int c = 0; c < s4_nchunks; ++c)
    {
        vector<vector<double>> Wc(num_col, vector<double>(s4_chunk));
        for (int r = 0; r < num_col; ++r)
            for (int q = 0; q < s4_chunk; ++q)
                Wc[r][q] = inter_weight[r][c*s4_chunk + q];
        vector<PhantomCiphertext> chunk = ct_pt_matrix_mul_wo_pre_large(rtn, Wc, num_col, s4_chunk, num_col, context);
        for (int q = 0; q < s4_chunk; ++q)
        {
            int idx = c*s4_chunk + q;
            PhantomPlaintext ecd_inter_bias;
            vector<double> inter_bias_vec(slot_count, 0);
            for (int j = 0; j < (int)slot_count; ++j){
                if(b_vec[j] == 1){
                    inter_bias_vec[j] = inter_bias[idx];
                }
            }
            encoder.encode(inter_bias_vec, chunk[q].params_id(), chunk[q].scale(), ecd_inter_bias);
            evaluator.mod_switch_to_inplace(ecd_inter_bias, chunk[q].params_id());
            chunk[q].scale() = scale;
            ecd_inter_bias.scale() = scale;
            evaluator.add_plain_inplace(chunk[q], ecd_inter_bias);
            inter_output_host[idx] = offload_cipher_to_host(chunk[q]);
        }
    }
    cudaDeviceSynchronize();
    std::cerr << "LEVER|chunk_ffn|slices=" << s4_nchunks << "|parked_ciphertexts="
              << inter_output_host.size() << "|disk="
              << (std::getenv("MOAI_SPILL_DIR") ? 1 : 0) << std::endl;
    vector<PhantomCiphertext> inter_output;   // s4_chunk_ffn: kept empty; the output lives on the host
"""

# 2. THE SEPARATE BIAS PASS. It is now done inside the slice loop, so the whole loop body goes.
#
# MATCHED BY REGEX, NOT BY LITERAL, and that is deliberate. The literal version of this anchor
# failed on the first attempt over invisible differences: four trailing spaces after
# `schedule(static)` and one after `bridge_to_default(stream);`. Trailing whitespace embedded in a
# patch is unreadable and breaks again on the next reformat, so the anchor tolerates it and pins the
# things that actually identify the loop instead -- the pragma, the `num_inter` bound, and the
# `add_plain_inplace` on `inter_output`.
BIAS_RE = re.compile(
    r"#pragma omp for schedule\(static\)[ \t]*\n"
    r"[ \t]*for \(int i = 0; i < num_inter; \+\+i\)\{.*?"
    r"evaluator_local\.add_plain_inplace\(inter_output\[i\],ecd_inter_bias\);\s*\n"
    r"[ \t]*\}\n",
    re.S)

BIAS_PATCH = """        // s4_chunk_ffn: the bias is added per slice inside the chunked matmul above, so this pass
        // has nothing left to do. The surrounding parallel region is kept so the stream pool and
        // the local encoder/evaluator are constructed exactly as before.
"""

# 3. THE GELU. Its input now comes back from the host one ciphertext at a time.
GELU_ANCHOR = """                gelu_output[i*32+j] = gelu_v2(inter_output[i*32+j],context,relin_keys,secret_key, stream);
"""

GELU_PATCH = """                // s4_chunk_ffn: the FFN output lives on the host; reload just this one.
                PhantomCiphertext _gin;
                reload_cipher_from_host(_gin, inter_output_host[i*32+j]);
                gelu_output[i*32+j] = gelu_v2(_gin,context,relin_keys,secret_key, stream);
"""

# 3b. THE LEVEL PRINT. It is not decoration: these `Modulus chain index ...` lines ARE the T1s gate,
# compared byte for byte, so the step has to reproduce this one exactly. The retired `w5` read the
# value out of the host copy's metadata; this reloads ONE ciphertext and asks the same question of
# the same object, which cannot print something different for a reason nobody checked.
PRINT_ANCHOR = """    cout <<"Modulus chain index after inter layer: "<< context.get_context_data(inter_output[0].params_id()).chain_depth()<<endl;
"""

PRINT_PATCH = """    // s4_chunk_ffn: the FFN output lives on the host, so reload one ciphertext to ask it the same
    // question. This line is part of the T1s gate and must print exactly what it printed before.
    {
        PhantomCiphertext _probe;
        reload_cipher_from_host(_probe, inter_output_host[0]);
        cout <<"Modulus chain index after inter layer: "<< context.get_context_data(_probe.params_id()).chain_depth()<<endl;
    }
"""

# 4. THE FREE. `inter_output` is empty now; the host bank is what has to be released.
SWAP_ANCHOR = """    vector<PhantomCiphertext>().swap(inter_output);
"""

SWAP_PATCH = """    vector<PhantomCiphertext>().swap(inter_output);
    std::vector<HostCipher>().swap(inter_output_host);   // s4_chunk_ffn: the GELU has consumed it
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "s4_chunk_ffn" in txt:
        sys.exit("FAILED: this tree already carries the lever")

    # THE PARK HELPERS MUST ALREADY BE IN THE TREE. They are `impl/borrowed/wpark_helpers.cuh`,
    # inserted by `patch_park_input_copy.py`; this step calls `offload_cipher_to_host` and
    # `reload_cipher_from_host` and cannot supply them itself.
    inc = os.path.join(sys.argv[1], "src/include/include.cuh")
    itxt = open(inc).read() if os.path.exists(inc) else ""
    for need in ("offload_cipher_to_host", "reload_cipher_from_host", "struct HostCipher"):
        if need not in itxt:
            sys.exit(f"FAILED: include.cuh does not carry `{need}`; apply the park helpers first "
                     "(patch_park_input_copy.py inserts impl/borrowed/wpark_helpers.cuh)")

    for name, anchor in (("matmul", MM_ANCHOR), ("gelu call", GELU_ANCHOR),
                         ("level print", PRINT_ANCHOR), ("inter_output swap", SWAP_ANCHOR)):
        if txt.count(anchor) != 1:
            sys.exit(f"FAILED: the {name} matched {txt.count(anchor)} times, expected exactly 1")
    _bias = BIAS_RE.findall(txt)
    if len(_bias) != 1:
        sys.exit(f"FAILED: the bias pass matched {len(_bias)} times, expected exactly 1")

    # THE SLICING MUST DIVIDE THE WIDTH, checked against the source rather than assumed: a remainder
    # would silently drop columns and the gate would move.
    m = re.search(r"num_inter\s*=\s*(\d+)", txt)
    if m and int(m.group(1)) % 768:
        sys.exit(f"FAILED: num_inter is {m.group(1)}, which 768 does not divide; the slice loop "
                 "would drop the remainder")

    # AND THE ORDER MUST BE matmul -> bias -> gelu -> swap, so the replacements cannot cross.
    pos = [txt.index(MM_ANCHOR), BIAS_RE.search(txt).start(),
           txt.index(GELU_ANCHOR), txt.index(SWAP_ANCHOR)]
    if pos != sorted(pos):
        sys.exit("FAILED: the four sites are not in the order this step assumes")

    txt = txt.replace(MM_ANCHOR, MM_PATCH, 1)
    txt = BIAS_RE.sub(lambda m: BIAS_PATCH, txt, count=1)
    txt = txt.replace(GELU_ANCHOR, GELU_PATCH, 1)
    txt = txt.replace(PRINT_ANCHOR, PRINT_PATCH, 1)
    txt = txt.replace(SWAP_ANCHOR, SWAP_PATCH, 1)

    # NOTHING MAY STILL SUBSCRIPT `inter_output`, which this step leaves empty. BOTH comment forms
    # are stripped first: the other patches on this line strip only `//`, which is enough where they
    # act, but the FFN region carries a `/* ... */` block holding a disabled decrypt-and-print loop
    # that subscripts `inter_output` four times. Counting those would refuse a correct patch.
    stripped = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    stripped = re.sub(r"//[^\n]*", "", stripped)
    left = re.findall(r"inter_output\s*\[", stripped)
    if left:
        sys.exit(f"FAILED: `inter_output[...]` is still read {len(left)} time(s), but this step "
                 "leaves that vector empty")

    open(src, "w").write(txt)
    print("patch_chunk_ffn: FFN computed in 4 slices of 768 and parked off the device (" + SRC + ")")


if __name__ == "__main__":
    main()
