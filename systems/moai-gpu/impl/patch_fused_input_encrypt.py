#!/usr/bin/env python3
"""MOAI-GPU s7: input never exists as a full-level array. Paid.

    patch_fused_input_encrypt.py <moai source tree>

batch_input's per-ciphertext body inlined into the existing loop: pack/encode/encrypt at full level,
residual derivation, park and mod-switch-down per ciphertext; one full-level ciphertext at a time.
Target: before_attention reserves 67,552 MiB (keys 40,498 + 27,648 MiB transient of 768 full-level
cts) with 52,018 live. mod_switch deterministic: encryption-draw order kept (CKKS: different valid
encryption of the same plaintext under the same key). Paid (input_x's swap after the loop holds
100.7 MB host, 4.6% of the host peak). Expected before_attention ~52,600, peak -17% to -22%
(62.5-66 GiB vs 80.2). Gate T1s (detects a wrong level, not a value change).
"""

import os
import sys

SRC = "src/include/test/test_single_layer.cuh"

OLD = """    // encode + encrypt
    vector<PhantomCiphertext> enc_ecd_x = batch_input(input_x, num_X, num_row, num_col, scale, context, public_key);
    vector<int> input_len(num_X, 0);
    input_len[0] = 11;
    vector<int> b_vec = bias_vec(input_len, num_X, num_row);
    // cout <<"Matrix X size = "<<num_row <<" * "<<num_col<<endl;
    // cout <<"Modulus chain index for x: "<< context.get_context_data(enc_ecd_x[0].parms_id())->chain_index()<<endl;

    vector<vector<vector<double>>>().swap(input_x);

    // s6_fused_park_input: the residual never exists as a device array. Each element is copied,
    // shrunk and parked before the next one is touched, so the device holds ONE ciphertext of it
    // at a time instead of 768. Same values, same per-element order, same mod-switch count; only
    // the lifetime of the array changed. `enc_ecd_x` is untouched until the loops below, so every
    // copy is still taken at exactly the level it had before.
    std::vector<HostCipher> enc_ecd_x_copy_host;
    enc_ecd_x_copy_host.reserve(num_col);
    for (int i = 0; i < num_col; ++i){
        PhantomCiphertext _one = enc_ecd_x[i];
        for (int j = 0; j < boot_level; ++j){
            evaluator.mod_switch_to_next_inplace(_one);
        }
        enc_ecd_x_copy_host.push_back(offload_cipher_to_host(_one));
    }
    cudaDeviceSynchronize();
    std::cerr << "LEVER|fused_park_input|parked_ciphertexts=" << enc_ecd_x_copy_host.size()
              << "|disk=" << (std::getenv("MOAI_SPILL_DIR") ? 1 : 0) << std::endl;

    // mod switch to remaining level
    // #pragma omp parallel for

    for (int i = 0; i < num_col; ++i)
    {
        for (int j = 0; j < boot_level + (remaining_level - remaining_level_att); ++j)
        {
            evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
        }
    }

    // mod switch to next level
    // #pragma omp parallel for

    for (int i = 0; i < num_col; ++i)
    {
        evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
    }
"""

NEW = """    // s7_fused_input_encrypt: the input never exists as a full-level array.
    //
    // `batch_input` filled a vector<PhantomCiphertext>(num_col) in a serial loop and RETURNED it,
    // so all 768 ciphertexts existed at FULL level before the caller switched anything down:
    // 768 x 36 MiB = 27,648, and keys 40,498 + 27,648 = the 67,552 MiB this phase reserved while
    // only 52,018 was ever live at the marker. That 15,534 MiB is a transient, not data.
    //
    // Its per-ciphertext body is inlined here so that everything which happens to a ciphertext
    // happens before the next one is created. Only ONE full-level ciphertext exists at any instant.
    // Borrowed from the retired campaign's `w7_fused_input`, which measured this phase 67,552 ->
    // 52,576 on the same figure.
    //
    // The encoder, encryptor and slot_count are the ones this function already constructed above,
    // from the same context and the same public key that were being passed to batch_input.
    // mod_switch is deterministic and consumes no randomness, so interleaving it between
    // encryptions leaves the order of encryption draws exactly as it was.
    vector<PhantomCiphertext> enc_ecd_x(num_col);
    vector<int> input_len(num_X, 0);
    input_len[0] = 11;
    vector<int> b_vec = bias_vec(input_len, num_X, num_row);

    std::vector<HostCipher> enc_ecd_x_copy_host;
    enc_ecd_x_copy_host.reserve(num_col);
    for (int i = 0; i < num_col; ++i)
    {
        // batch_input's body, verbatim from source/matrix_mul/Batch_encode_encrypt.cuh
        vector<double> _fie_vec(slot_count, 0);
        for (int j = 0; j < num_X; ++j)
        {
            for (int k = 0; k < num_row; ++k)
            {
                _fie_vec[num_X * k + j] = input_x[j][k][i];
            }
        }
        PhantomPlaintext _fie_ecd_vec;
        encoder.encode(_fie_vec, scale, _fie_ecd_vec);
        encryptor.encrypt(_fie_ecd_vec, enc_ecd_x[i]);

        // s6_fused_park_input's body, unchanged: the LN1 residual copy at boot level, parked.
        PhantomCiphertext _one = enc_ecd_x[i];
        for (int j = 0; j < boot_level; ++j){
            evaluator.mod_switch_to_next_inplace(_one);
        }
        enc_ecd_x_copy_host.push_back(offload_cipher_to_host(_one));

        // the two mod-switch loops' bodies, unchanged, applied to this ciphertext
        for (int j = 0; j < boot_level + (remaining_level - remaining_level_att); ++j)
        {
            evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
        }
        evaluator.mod_switch_to_next_inplace(enc_ecd_x[i]);
    }
    cudaDeviceSynchronize();
    // THE SWAP MOVED. It used to run before the loop; the loop reads input_x, so it runs after.
    // That holds 128 x 128 x 768 doubles = 100.7 MB of HOST memory for the duration of input
    // preparation, which is 4.6% of this system's host peak and must show in the measurement.
    vector<vector<vector<double>>>().swap(input_x);
    std::cerr << "LEVER|fused_input_encrypt|ciphertexts=" << num_col
              << "|parked=" << enc_ecd_x_copy_host.size()
              << "|disk=" << (std::getenv("MOAI_SPILL_DIR") ? 1 : 0) << std::endl;
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tree = sys.argv[1]
    src = os.path.join(tree, SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + src)
    txt = open(src).read()

    if "s7_fused_input_encrypt" in txt:
        sys.exit("FAILED: this tree already carries the lever")

    # The parent is checked: this step's OLD block IS s6's output, so a tree without s6's park loop
    # cannot match (the two failures need different fixes).
    if "s6_fused_park_input" not in txt:
        sys.exit("FAILED: this tree does not carry patch_park_input_copy's fused park loop; this "
                 "step rewrites that loop and must be applied on top of it")
    if "offload_cipher_to_host" not in txt:
        sys.exit("FAILED: this tree does not carry the park helpers")

    if txt.count(OLD) != 1:
        sys.exit(f"FAILED: the input-preparation block matched {txt.count(OLD)} times, expected "
                 f"exactly 1; the tree is not the one this patch was written against")

    txt = txt.replace(OLD, NEW, 1)

    # batch_input must be gone from this function, checked scoped to it (other test drivers in the
    # tree call it legitimately). Guards a second call left behind rebuilding the full-level array.
    fn = txt.split("void single_layer_test", 1)
    if len(fn) != 2:
        sys.exit("FAILED: cannot locate single_layer_test to scope the batch_input check")
    if "batch_input(" in fn[1].split("\nvoid ", 1)[0]:
        sys.exit("FAILED: single_layer_test still calls batch_input after the rewrite; the "
                 "full-level array would be rebuilt and this step would measure nothing")

    open(src, "w").write(txt)
    print("patch_fused_input_encrypt: batch_input is inlined per ciphertext; only one full-level "
          "ciphertext exists at a time")


if __name__ == "__main__":
    main()
