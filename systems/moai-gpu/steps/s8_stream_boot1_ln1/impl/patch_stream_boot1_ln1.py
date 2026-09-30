#!/usr/bin/env python3
"""MOAI-GPU s8: bootstrap 1 streams its output; LayerNorm1 reads it from the host. Paid.

    patch_stream_boot1_ln1.py <moai source tree>

Basis: s7_stream_local_reload series:
    phase                 used     reserved     growth     gap
    before_attention    52,018       67,552    +26,976  15,534
    after_attention     53,554       69,984     +2,432  16,430
    after_bootstrap1    58,162       69,984         +0  11,822
    after_layernorm1    58,930       79,488     +9,504  20,558   <- THIS STEP
    after_bootstrap2    49,714       79,488         +0
    after_intermediate  49,714       79,488         +0
    after_gelu          55,858       79,488         +0
    after_bootstrap3    55,858       79,488         +0
    after_layernorm2    56,626       79,488         +0
    after_bootstrap4    64,306       81,248     +1,760  16,942
LayerNorm1 reserves 9,504 MiB for 768 MiB more data (LN1 needs its 768-ct array across three passes).
Not input preparation: s7_fused_input_encrypt cut before_attention 67,552 -> 52,608 MiB, peak +0.04%
(attention grew +17,824), peak_alloc_kb unchanged.
Mechanism (s6_stream_boot23, -22.42%):
  * bootstrap 1 writes each output to the host (rtn never holds 768);
  * LN1 residual add folded into that loop (replaces s5_park_input_copy's reload pass);
  * layernorm_streaming reads inputs one at a time across three passes.
Donor: retired s9_streaming_ln1 ("boot1 live 85042 -> 42034 (= just the key bank)");
borrowed/layernorm1_streaming.cuh = 184 lines verbatim; differs from layernorm2 in two constants
(variance normaliser 1/768^2 vs 1/768^3; gamma/sqrt(768) vs gamma/768); v1_verify maxabsdiff = 0.
Cost: 768 more D2H at bootstrap 1; LN1 reloads three times (s6: +1.61% for two arrays).
Gate: expected identical (md5 01469da3...); levels unchanged.
Expected: peak 71,000 to 73,000 MiB vs 81,248 (-10% to -13%); else the LN1 reservation is the pool's
response to the previous phase (allocator trace next).
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "src/include/test/test_single_layer.cuh"
LN = "src/include/source/non_linear_func/layernorm.cuh"
LNS = os.path.join(HERE, "borrowed", "layernorm1_streaming.cuh")

BOOT1_OLD = """    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn[i*6+j],att_selfoutput[i*6+j]);
            // s2 boot_input_release: this input is dead the instant its own bootstrap
            // returns; the stock code holds all 768 until after the loop.
            att_selfoutput[i*6+j] = PhantomCiphertext();
            if (i == 0 && j == 0) std::cerr << "LEVER|boot_input_release|sites=4" << std::endl;
        }
    }

    gettimeofday(&tend1,NULL);
    double boot_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time<<endl;
    append_csv_row("../single_layer_results.csv", "1st Bootstrapping", boot_time);
    cout <<"Modulus chain index after bootstrapping: "<< context.get_context_data(rtn[0].params_id()).chain_depth()<<endl;
    snni_mem_marker("after_bootstrap1");
"""

BOOT1_NEW = """    // s8_stream_boot1_ln1: bootstrap 1 streams its output to the host as each ciphertext is
    // produced, and the LN1 residual add is folded in here. The residual is ALREADY on the host
    // (s5_park_input_copy put it there), so reloading it inside this loop REPLACES the separate
    // pass that used to run below rather than adding a round trip. `rtn` never holds 768 at once.
    std::vector<HostCipher> rtn1_host(att_selfoutput_size);
    size_t _boot1_chain = 0;
    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            int _idx = i*6+j;
            PhantomCiphertext _c;
            bootstrapper.bootstrap_3(_c, att_selfoutput[_idx]);
            // s2 boot_input_release, kept: the input is dead the instant its bootstrap returns.
            att_selfoutput[_idx] = PhantomCiphertext();
            if (_idx == 0) std::cerr << "LEVER|boot_input_release|sites=4" << std::endl;
            // the LN1 residual add, moved here from the separate pass below. Same operands, same
            // per-element order, same mod-switch target.
            PhantomCiphertext _res;
            reload_cipher_from_host(_res, enc_ecd_x_copy_host[_idx]);
            evaluator.mod_switch_to_inplace(_res, _c.params_id());
            evaluator.add_inplace(_c, _res);
            if (_idx == 0) _boot1_chain = context.get_context_data(_c.params_id()).chain_depth();
            rtn1_host[_idx] = offload_cipher_to_host(_c);
        }
    }
    cudaDeviceSynchronize();
    std::vector<HostCipher>().swap(enc_ecd_x_copy_host);   // residual fully consumed
    // `rtn` IS DELIBERATELY LEFT ALONE, and the first version of this patch got it wrong. Emptying
    // it looks right -- boot1 no longer writes it -- but BOOTSTRAP 2 writes `rtn[i*6+j]` further
    // down, so a swap here would index past the end of an empty vector. It keeps the 768
    // default-constructed ciphertexts it held before boot1 ran, which cost nothing, and boot2 fills
    // them exactly as before. The donor's `s9_streaming_ln1` leaves it alone for the same reason.
    std::cerr << "LEVER|stream_boot1_ln1|parked=" << rtn1_host.size() << std::endl;

    gettimeofday(&tend1,NULL);
    double boot_time = tend1.tv_sec-tstart1.tv_sec+(tend1.tv_usec-tstart1.tv_usec)/1000000.0;
    cout <<"bootstrapping time = "<<boot_time<<endl;
    append_csv_row("../single_layer_results.csv", "1st Bootstrapping", boot_time);
    cout <<"Modulus chain index after bootstrapping: "<< _boot1_chain <<endl;
    snni_mem_marker("after_bootstrap1");
"""

ADD_OLD = """    // s5_park_input_copy: reload ONE ciphertext at a time, so the residual is never fully device
    // resident again. Same arithmetic in the same order; only the source of each operand changed.
    for (int i = 0; i < num_col; ++i){
        PhantomCiphertext _res;
        reload_cipher_from_host(_res, enc_ecd_x_copy_host[i]);
        evaluator.mod_switch_to_inplace(_res, rtn[i].params_id());
        evaluator.add_inplace(rtn[i], _res);
    }
    std::vector<HostCipher>().swap(enc_ecd_x_copy_host);

    vector<PhantomCiphertext> layernorm_selfoutput = layernorm(rtn,layernorm1_gamma,layernorm1_beta, b_vec,
        context,relin_keys,secret_key);
"""

ADD_NEW = """    // s8_stream_boot1_ln1: the residual add was folded into the streaming bootstrap loop above,
    // per ciphertext, before each output was offloaded. LayerNorm1 now reads its inputs from that
    // host bank one at a time across its three passes.
    vector<PhantomCiphertext> layernorm_selfoutput = layernorm_streaming(rtn1_host,layernorm1_gamma,layernorm1_beta, b_vec,
        context,relin_keys,secret_key);
    std::vector<HostCipher>().swap(rtn1_host);   // free the host bank once LN1 has consumed it
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tree = sys.argv[1]
    src = os.path.join(tree, SRC)
    ln = os.path.join(tree, LN)
    for p in (src, ln):
        if not os.path.exists(p):
            sys.exit("FAILED: no " + p)
    if not os.path.exists(LNS):
        sys.exit("FAILED: no borrowed/layernorm1_streaming.cuh beside this patch; the mechanism is "
                 "taken verbatim from the retired image and is not reconstructed here")

    txt = open(src).read()
    lntxt = open(ln).read()

    if "s8_stream_boot1_ln1" in txt:
        sys.exit("FAILED: this tree already carries the lever")

    # THE PARENTS ARE CHECKED, because this step rewrites what two earlier ones wrote. Saying which
    # one is missing beats a bare anchor miss: the fixes differ.
    if "s5_park_input_copy" not in txt:
        sys.exit("FAILED: this tree does not carry patch_park_input_copy; the LN1 residual is not "
                 "on the host and this step's folded add has nothing to reload")
    if "layernorm2_streaming" not in lntxt:
        sys.exit("FAILED: this tree does not carry patch_stream_boot23; applying LN1 streaming "
                 "before LN2 would measure against the wrong peak")
    if "offload_cipher_to_host" not in lntxt and "offload_cipher_to_host" not in txt:
        sys.exit("FAILED: this tree does not carry the park helpers")

    # ---- the borrowed streaming LayerNorm1 ----
    if "layernorm_streaming" not in lntxt:
        block = open(LNS).read()
        if "layernorm_streaming" not in block:
            sys.exit("FAILED: borrowed/layernorm1_streaming.cuh does not define layernorm_streaming")
        lntxt = lntxt.rstrip("\n") + "\n" + block
        open(ln, "w").write(lntxt)
        print(f"  layernorm_streaming appended to {LN} ({block.count(chr(10))} borrowed lines)")

    for name, old, new in (("boot1 loop", BOOT1_OLD, BOOT1_NEW),
                           ("residual add + layernorm1 call", ADD_OLD, ADD_NEW)):
        n = txt.count(old)
        if n != 1:
            sys.exit(f"FAILED: anchor `{name}` matched {n} times, expected exactly 1")
        txt = txt.replace(old, new, 1)

    # `rtn` MUST NOT BE READ BETWEEN THE BOOTSTRAP AND LAYERNORM2, and this is the check that
    # matters: the step empties it, so a surviving read would decrypt an uninitialised ciphertext.
    # Comments are stripped first, both kinds -- this tree carries a large `/* */` block holding a
    # stock decrypt-and-print loop that names `rtn[i]`, and a check reading prose as code has
    # refused a correct patch on this system twice.
    after = txt.split('snni_mem_marker("after_bootstrap1");', 1)
    if len(after) != 2:
        sys.exit("FAILED: no after_bootstrap1 marker to scope the dead-read check")
    # THE WINDOW ENDS AT THE LAYERNORM1 CALL, not at the next marker. `rtn` is legitimately REUSED
    # by bootstrap 2 further down -- it is the same array serving a second purpose -- so a window
    # reaching that far reports a correct patch as broken. The first version of this check did
    # exactly that, and the report was confusing precisely because the thing it named was fine.
    tail = after[1].split("layernorm_streaming(rtn1_host", 1)[0]
    live = re.sub(r"/\*.*?\*/", "", tail, flags=re.S)
    live = re.sub(r"//[^\n]*", "", live)
    if re.search(r"\brtn\s*\[", live):
        sys.exit("FAILED: `rtn` is still indexed between bootstrap 1 and the LayerNorm1 call; "
                 "boot1 no longer writes it there, so a read would use an uninitialised ciphertext")

    open(src, "w").write(txt)
    print("patch_stream_boot1_ln1: boot1 streams to the host, LN1 reads it back, residual folded in")


if __name__ == "__main__":
    main()
