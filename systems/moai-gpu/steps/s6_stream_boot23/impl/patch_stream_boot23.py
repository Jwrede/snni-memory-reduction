#!/usr/bin/env python3
"""MOAI-GPU: bootstraps 2 and 3 stream per ciphertext (no 768-wide array). Paid.

    patch_stream_boot23.py <moai source tree>

Donor code (not reimplemented): retired w2_stream_boot_layer and w3_stream_boot4 images; their
test_single_layer.cuh and layernorm.cuh diffed against w1_park_residuals.
borrowed/layernorm2_streaming.cuh = 186 lines verbatim from w3_stream_boot4's layernorm.cuh.
    w1_park_residuals    115.44 GiB
    w2_stream_boot_layer 102.82        <- this patch's first half
    w3_stream_boot4       92.82        <- this patch's second half
BOOT2: boot_layer (full-level deep copy of each bootstrap output for the LN2 residual) offloaded per
element; rtn's eleven mod_switch_to_next_inplace calls moved into the loop (else rtn holds 768
full-level cts, ~42 GiB); the residual copy already holds the full-level value.
BOOT3: same; LN2 residual add folded in (bootstrap, free input, reload residual, mod-switch, add,
park); rtn2 never on device. LayerNorm2 reads the host bank via layernorm2_streaming (three passes).
Basis: s5_park_input_chunk_ffn:
    before_attention    67,552 MiB      fused_park_input works
    after_gelu          96,544          chunk_ffn works
    after_bootstrap3   105,888          <- the peak, and nothing after it grows
p_bootprobe: +20.0 MiB per call, no retained workspace (growth = output array).
Deviations from the donor (harness):
  1. gate: the after-boot2 `Modulus chain index` line kept (depth captured from element 0 before
     offload); same value, position, line count.
  2. print_cuda_meminfo -> snni_mem_marker.
  3. s2_boot_input_release kept in the boot2 loop.
Cost: 768 more D2H at boot2 and boot3; LayerNorm2 reloads three times. Wall measured here.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LNS = os.path.join(HERE, "borrowed", "layernorm2_streaming.cuh")
SRC = "src/include/test/test_single_layer.cuh"
LN = "src/include/source/non_linear_func/layernorm.cuh"

# ---------------------------------------------------------------------------------------------
# BOOT2. Three edits, anchored on text this tree is known to carry.

B2_DECL_OLD = "    vector<PhantomCiphertext> boot_layer(layernorm_selfoutput_size);\n"
B2_DECL_NEW = (
    "    vector<PhantomCiphertext> boot_layer(layernorm_selfoutput_size);\n"
    "    // w2: the LN2 residual copy is streamed to the host INSIDE the boot2 loop, so boot_layer\n"
    "    // never fully materialises on the device co-resident with rtn.\n"
    "    std::vector<HostCipher> boot_layer_host(layernorm_selfoutput_size);\n"
    "    size_t _boot2_chain = 0;\n"
)

B2_LOOP_OLD = """            boot_layer[i*6+j] = rtn[i*6+j];
        }
    }

    // #pragma omp parallel for

    for (int i = 0; i < rtn.size(); ++i) {
        for (int j = 0; j < 11; ++j){
            evaluator.mod_switch_to_next_inplace(rtn[i]);
        }
    }
"""
B2_LOOP_NEW = """            boot_layer[i*6+j] = rtn[i*6+j];                                      // deep device copy
            // THE GATE LINE THIS TREE PRINTS AFTER BOOTSTRAP 2 reads boot_layer[0], which is about
            // to leave the device. The depth is captured here so the printed sequence is unchanged.
            if (i == 0 && j == 0)
                _boot2_chain = context.get_context_data(boot_layer[0].params_id()).chain_depth();
            boot_layer_host[i*6+j] = offload_cipher_to_host(boot_layer[i*6+j]);   // D2H + free
            // w2: drop rtn to its consumer level RIGHT HERE. As a second pass after the loop this
            // let rtn accumulate 768 FULL-level ciphertexts first. The residual copy above already
            // captured the full-level value, so this is a reorder of deterministic per-ciphertext
            // operations and is numerically identical.
            for (int m = 0; m < 11; ++m){
                evaluator.mod_switch_to_next_inplace(rtn[i*6+j]);
            }
        }
    }
    cudaDeviceSynchronize();
    vector<PhantomCiphertext>().swap(boot_layer);
"""

B2_PRINT_OLD = (
    '    cout <<"Modulus chain index after bootstrapping: "'
    "<< context.get_context_data(boot_layer[0].params_id()).chain_depth()<<endl;\n"
)
B2_PRINT_NEW = (
    '    cout <<"Modulus chain index after bootstrapping: "<< _boot2_chain <<endl;\n'
)

# ---------------------------------------------------------------------------------------------
# BOOT3. Two edits: the loop, and the residual-add pass that the loop absorbs.

B3_LOOP_OLD = """    vector<PhantomCiphertext> rtn2(768);
    gettimeofday(&tstart1,NULL);

    // #pragma omp parallel for

    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn2[i*6+j],final_output[i*6+j]);
            // s2 boot_input_release: this input is dead the instant its own bootstrap
            // returns; the stock code holds all 768 until after the loop.
            final_output[i*6+j] = PhantomCiphertext();
        }
    }
"""
B3_LOOP_NEW = """    // w3: bootstrap ONE ciphertext at a time, fold in the LN2 residual add, park the result on
    // the host. rtn2 is never a device-resident 768-ciphertext array.
    vector<PhantomCiphertext> rtn2;                 // stays empty here; reused by the 4th bootstrap
    std::vector<HostCipher> rtn2_host(768);
    size_t _boot3_chain = 0;
    gettimeofday(&tstart1,NULL);

    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            int _idx = i*6+j;
            PhantomCiphertext _c;
            bootstrapper.bootstrap_3(_c, final_output[_idx]);
            // s2 boot_input_release, kept: this input is dead the instant its own bootstrap returns.
            final_output[_idx] = PhantomCiphertext();
            PhantomCiphertext _res;                 // the LN2 residual add, folded in from below
            reload_cipher_from_host(_res, boot_layer_host[_idx]);
            evaluator.mod_switch_to_inplace(_res, _c.params_id());
            evaluator.add_inplace(_c, _res);
            if (_idx == 0) _boot3_chain = context.get_context_data(_c.params_id()).chain_depth();
            rtn2_host[_idx] = offload_cipher_to_host(_c);
        }
    }
    cudaDeviceSynchronize();
    std::vector<HostCipher>().swap(boot_layer_host);   // the residual is fully consumed
"""

# ANCHORED TOGETHER WITH ITS MARKER. The same print appears again after bootstrap 4, which also
# reads `rtn2[0]`, so the line alone matches twice and the patch refused rather than guessing.
B3_PRINT_OLD = (
    '    cout <<"Modulus chain index after bootstrapping: "'
    "<< context.get_context_data(rtn2[0].params_id()).chain_depth()<<endl;\n"
    '    snni_mem_marker("after_bootstrap3");\n'
)
B3_PRINT_NEW = (
    '    cout <<"Modulus chain index after bootstrapping: "<< _boot3_chain <<endl;\n'
    '    snni_mem_marker("after_bootstrap3");\n'
)

# BOOTSTRAP 4 IS DELIBERATELY NOT TOUCHED, although the donor streams it too (its `s7`). Our own
# markers say it does not grow the pool: `s5_park_input_chunk_ffn` reserves 105,888 MiB at
# `after_bootstrap3` and the SAME 105,888 at `after_bootstrap4`. A lever there cannot lower a peak
# it does not set. If this patch moves the peak onto boot4, that is the next rung and it will have
# its own measurement to stand on.

# The residual-add pass the boot3 loop has absorbed. Matched by regex because this tree's copy
# carries `s3_boot_layer_release`'s comment block, whose wording must not be depended on.
ADD_PASS_RE = re.compile(
    r"    for \(int i = 0; i < num_col; \+\+i\)\{\n"
    r"        evaluator\.mod_switch_to_inplace\(boot_layer\[i\], rtn2\[i\]\.params_id\(\)\);\n"
    r"        evaluator\.add_inplace\(rtn2\[i\],boot_layer\[i\]\);\n"
    r"(?:.*\n)*?"
    r"        boot_layer\[i\] = PhantomCiphertext\(\);\n"
    r"    \}\n"
)
ADD_PASS_NEW = (
    "    // w3: this pass is gone. Its mod-switch and add now happen inside the boot3 loop above,\n"
    "    // per ciphertext, before the result is parked -- which is what lets rtn2 stay off the\n"
    "    // device. `s3_boot_layer_release`'s per-element free is subsumed: boot_layer_host is\n"
    "    // swapped out in one go once the loop has consumed it.\n"
)

LN2_CALL_OLD = """    vector<PhantomCiphertext> layernorm_finaloutput = layernorm2(rtn2,layernorm2_gamma,layernorm2_beta,b_vec,
        context,relin_keys,secret_key);
"""
LN2_CALL_NEW = """    // w3: streaming LayerNorm2 reads its 768 inputs from the HOST, one at a time, in three passes.
    vector<PhantomCiphertext> layernorm_finaloutput = layernorm2_streaming(rtn2_host,layernorm2_gamma,layernorm2_beta,b_vec,
        context,relin_keys,secret_key);
    std::vector<HostCipher>().swap(rtn2_host);   // freed once LN2 has consumed it
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tree = sys.argv[1]
    src = os.path.join(tree, SRC)
    ln = os.path.join(tree, LN)
    for p in (src, ln, LNS):
        if not os.path.exists(p):
            sys.exit("FAILED: no " + p)

    txt = open(src).read()
    lntxt = open(ln).read()

    if "boot_layer_host" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    # The infrastructure is shared and checked separately, exactly as patch_park_input_copy does.
    for need in ("offload_cipher_to_host", "reload_cipher_from_host"):
        if need not in open(os.path.join(tree, "src/include/include.cuh")).read():
            sys.exit(f"FAILED: include.cuh does not carry `{need}`; run patch_park_helpers.py first")

    # ---- the borrowed streaming LayerNorm2 ----
    if "layernorm2_streaming" not in lntxt:
        block = open(LNS).read()
        if "layernorm2_streaming" not in block:
            sys.exit("FAILED: borrowed/layernorm2_streaming.cuh does not define layernorm2_streaming")
        lntxt = lntxt.rstrip("\n") + "\n" + block
        open(ln, "w").write(lntxt)
        print(f"  layernorm2_streaming appended to {LN} ({block.count(chr(10))} borrowed lines)")

    # ---- the six edits, each asserted ----
    edits = [
        ("boot2 host bank", B2_DECL_OLD, B2_DECL_NEW),
        ("boot2 loop", B2_LOOP_OLD, B2_LOOP_NEW),
        ("boot2 gate line", B2_PRINT_OLD, B2_PRINT_NEW),
        ("boot3 loop", B3_LOOP_OLD, B3_LOOP_NEW),
        ("boot3 gate line", B3_PRINT_OLD, B3_PRINT_NEW),
        ("layernorm2 call", LN2_CALL_OLD, LN2_CALL_NEW),
    ]
    for name, old, new in edits:
        n = txt.count(old)
        if n != 1:
            sys.exit(f"FAILED: anchor `{name}` matched {n} times, expected exactly 1")
        txt = txt.replace(old, new, 1)

    m = ADD_PASS_RE.search(txt)
    if not m:
        sys.exit("FAILED: the LN2 residual-add pass did not match; it is what the boot3 loop absorbs "
                 "and leaving it in place would add the residual TWICE")
    txt = txt[:m.start()] + ADD_PASS_NEW + txt[m.end():]

    # NOTHING MAY STILL READ THE ARRAYS THAT ARE NOW EMPTY, and the check is scoped to where that
    # is true. `boot_layer` legitimately holds ONE element inside the boot2 loop, which is the whole
    # mechanism; what must not survive is any use AFTER the loop has swapped it out. The first
    # version of this check tested the whole file and refused the patch's own loop body.
    _after = txt.split('snni_mem_marker("after_bootstrap2");', 1)
    if len(_after) != 2:
        sys.exit("FAILED: no after_bootstrap2 marker; cannot scope the dead-read check")
    if "boot_layer[" in _after[1]:
        sys.exit("FAILED: `boot_layer` is still indexed after bootstrap 2; it no longer holds data "
                 "there, and reading it would add the LN2 residual from an empty ciphertext")

    open(src, "w").write(txt)
    print("patch_stream_boot23: bootstrap 2 and 3 stream per ciphertext; rtn2 never on the device")
    for name, _, _ in edits:
        print(f"  applied: {name}")
    print("  applied: LN2 residual-add pass absorbed into the boot3 loop")


if __name__ == "__main__":
    main()
