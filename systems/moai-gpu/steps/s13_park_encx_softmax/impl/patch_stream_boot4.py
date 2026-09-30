#!/usr/bin/env python3
"""MOAI-GPU: bootstrap 4 streams its output too. Paid.

    patch_stream_boot4.py <moai source tree>

Refuted against s6 (boot4 grew 0), applied against s8 (s7_stream_local_reload moved growth back to
boot4). after_bootstrap4 = live maximum (64,306 MiB = 40,498 keys + 23,808 = 768 full-level outputs).
Layer output streamed to the host per ciphertext (rtn2 empty, out_host holds it); the gate reloads
each element before decrypt (round trip proven by s4_chunk_ffn, gate md5
01469da37e952299bae882665f16a0e1). Donor: retired w3_stream_boot4. Paid (~+1% wall). Expected live
maximum back to after_final 56,626 MiB (-12.0%).
"""

import os
import sys

SRC = "src/include/test/test_single_layer.cuh"

BOOT4_OLD = """    rtn2 = vector<PhantomCiphertext>(layernorm2_size);
    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            bootstrapper.bootstrap_3(rtn2[i*6+j],layernorm_finaloutput[i*6+j]);
            // s2 boot_input_release: this input is dead the instant its own bootstrap
            // returns; the stock code holds all 768 until after the loop.
            layernorm_finaloutput[i*6+j] = PhantomCiphertext();
        }
    }
"""
BOOT4_NEW = """    // w3/s7: stream the fourth bootstrap's output -- the LAYER OUTPUT -- to the host as each
    // ciphertext is produced. Holding all 768 at full level is what makes boot4 the binding phase
    // once boot2 and boot3 no longer are.
    rtn2 = vector<PhantomCiphertext>();          // stays empty; the output lives on the host
    std::vector<HostCipher> out_host(layernorm2_size);
    size_t _boot4_chain = 0;
    for(int i = 0 ; i < 128 ; ++i){
        for(int j = 0 ; j < 6 ; ++j){
            int _idx = i*6+j;
            PhantomCiphertext _c;
            bootstrapper.bootstrap_3(_c, layernorm_finaloutput[_idx]);
            // s2 boot_input_release, kept: the input is dead the instant its bootstrap returns.
            layernorm_finaloutput[_idx] = PhantomCiphertext();
            if (_idx == 0) _boot4_chain = context.get_context_data(_c.params_id()).chain_depth();
            out_host[_idx] = offload_cipher_to_host(_c);
        }
    }
    cudaDeviceSynchronize();
"""

PRINT_OLD = ('    cout <<"Modulus chain index after bootstrapping: "'
             "<< context.get_context_data(rtn2[0].params_id()).chain_depth()<<endl;\n"
             '    snni_mem_marker("after_bootstrap4");\n')
PRINT_NEW = ('    cout <<"Modulus chain index after bootstrapping: "<< _boot4_chain <<endl;\n'
             '    snni_mem_marker("after_bootstrap4");\n')

# The gate loop decrypts every element of the layer output (now on the host), so it reloads each
# first; the observable is otherwise unchanged.
GATE_OLD ="        for (int i = 0; i < (int)rtn2.size(); ++i) {\n"
GATE_NEW = "        for (int i = 0; i < (int)out_host.size(); ++i) {\n"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tree = sys.argv[1]
    src = os.path.join(tree, SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + src)
    txt = open(src).read()

    if "out_host" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if "boot_layer_host" not in txt or "rtn2_host" not in txt:
        sys.exit("FAILED: this tree does not carry patch_stream_boot23; boot4 is only the binding "
                 "phase once boot2 and boot3 are not, and on a tree without them this step would "
                 "be measured against the wrong peak")

    for name, old in (("boot4 loop", BOOT4_OLD), ("boot4 gate line", PRINT_OLD),
                      ("gate decrypt loop", GATE_OLD)):
        if txt.count(old) != 1:
            sys.exit(f"FAILED: anchor `{name}` matched {txt.count(old)} times, expected exactly 1")

    txt = txt.replace(BOOT4_OLD, BOOT4_NEW, 1)
    txt = txt.replace(PRINT_OLD, PRINT_NEW, 1)
    txt = txt.replace(GATE_OLD, GATE_NEW, 1)

    # The gate body still names `rtn2[i]`; it must read the reloaded ciphertext instead.
    body_old = "            decryptor.decrypt(rtn2[i], g_pt);"
    body_new = ("            PhantomCiphertext _oc;\n"
                "            reload_cipher_from_host(_oc, out_host[i]);\n"
                "            decryptor.decrypt(_oc, g_pt);")
    if txt.count(body_old) != 1:
        sys.exit(f"FAILED: the gate's decrypt matched {txt.count(body_old)} times, expected 1")
    txt = txt.replace(body_old, body_new, 1)

    after = txt.split('snni_mem_marker("after_bootstrap4");', 1)
    if len(after) != 2:
        sys.exit("FAILED: no after_bootstrap4 marker to scope the dead-read check")
    # Both comment kinds stripped before the check: a large /* */ block after boot4 holds the stock
    # decrypt-and-print loop still naming rtn2[i].
    import re as _re
    live = _re.sub(r"/\*.*?\*/", "", after[1], flags=_re.S)
    live = _re.sub(r"//[^\n]*", "", live)
    if "rtn2[i]" in live or "rtn2[0]" in live:
        sys.exit("FAILED: `rtn2` is still indexed after bootstrap 4; it is empty there and a read "
                 "would decrypt an uninitialised ciphertext into the GATE")

    open(src, "w").write(txt)
    print("patch_stream_boot4: the layer output streams to the host; rtn2 never holds it")
    print("  applied: boot4 loop, boot4 gate line, gate decrypt loop")


if __name__ == "__main__":
    main()
