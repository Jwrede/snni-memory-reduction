#!/usr/bin/env python3
"""MOAI-GPU s11: enc_X parked during the softmax; enc_X_v freed after its last read (both inside the
attention block). Paid.

    patch_park_encx_softmax.py <moai source tree>

Context: s10_fused_input_encrypt: attention block ends at ~70,000 MiB reserved regardless of entry
(read as fragmentation, corrected). Levers freeing inside the allocating loop let blocks serve the
next request. Donor: w6_park_encx (diff vs w5_chunked_ffn = these two hunks). enc_X: last in-head use
derives enc_X_v; idle ~11.5 GiB through softmax_boot; parked, reloaded before return. enc_X_v (768
cts): V-matmul input only, freed after. Paid (768 D2H per head + 768 back). Expected after_attention
70,048 -> toward live 53,554; peak 62,000-68,000 MiB vs 78,624 (-14% to -21%). Gate T1s.
"""

import os
import sys

SRC = "src/include/source/att_block/single_att_block.cuh"

PARK_OLD = """    vector<PhantomCiphertext> V = ct_pt_matrix_mul_wo_pre(enc_X_v, WV, num_col, col_W, num_col, context);
"""

PARK_NEW = """    // s11: enc_X's last in-head use is deriving enc_X_v above (Q and K are already computed). It
    // then sits IDLE (~11.5 GiB) through softmax_boot's internal bootstraps -- the phase that binds
    // the whole run -- but the NEXT head needs it again. Park it on the host for the rest of this
    // head; reloaded before return.
    auto _enc_X_host = offload_vector_to_host(enc_X);
    vector<PhantomCiphertext> V = ct_pt_matrix_mul_wo_pre(enc_X_v, WV, num_col, col_W, num_col, context);
    // s11 (donor's s13): enc_X_v (768 cts) is only the input of the V matmul above -- dead from
    // here on, but it used to sit resident through softmax_boot, which bootstraps and therefore
    // sets the attention phase peak, until the function returned.
    vector<PhantomCiphertext>().swap(enc_X_v);
"""

RELOAD_OLD = """    vector<PhantomCiphertext> output = ct_ct_matrix_mul_diagpacking(QK, V, RotK, relin_keys,
                                                                    context, 128, 128, col_W, 128, num_batch);
"""

RELOAD_NEW = """    vector<PhantomCiphertext> output = ct_ct_matrix_mul_diagpacking(QK, V, RotK, relin_keys,
                                                                    context, 128, 128, col_W, 128, num_batch);

    // s11: restore enc_X for the caller (the next head reads it again).
    reload_vector_from_host(enc_X, _enc_X_host);
    std::vector<HostCipher>().swap(_enc_X_host);
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tree = sys.argv[1]
    src = os.path.join(tree, SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + src)
    txt = open(src).read()

    if "s11:" in txt or "_enc_X_host" in txt:
        sys.exit("FAILED: this tree already carries the lever")

    inc = os.path.join(tree, "src/include/include.cuh")
    if not os.path.exists(inc) or "offload_vector_to_host" not in open(inc).read():
        sys.exit("FAILED: this tree does not carry the park helpers (offload_vector_to_host); this "
                 "lever is applied on top of the steps that introduce them")

    for name, old in (("park site", PARK_OLD), ("reload site", RELOAD_OLD)):
        if txt.count(old) != 1:
            sys.exit(f"FAILED: anchor `{name}` matched {txt.count(old)} times, expected exactly 1")

    txt = txt.replace(PARK_OLD, PARK_NEW, 1)
    txt = txt.replace(RELOAD_OLD, RELOAD_NEW, 1)

    # The park must precede the reload: the anchors are 140 lines apart and a reversed tree would
    # compile and read an uninitialised host bank.
    if txt.index("_enc_X_host = offload_vector_to_host") > txt.index("reload_vector_from_host(enc_X"):
        sys.exit("FAILED: the reload site sits BEFORE the park site")

    # enc_X_v must not be read after it is emptied. Comments stripped first, both kinds.
    import re
    after = txt.split("vector<PhantomCiphertext>().swap(enc_X_v);", 1)[1]
    live = re.sub(r"/\*.*?\*/", "", after, flags=re.S)
    live = re.sub(r"//[^\n]*", "", live)
    if re.search(r"\benc_X_v\s*\[", live):
        sys.exit("FAILED: `enc_X_v` is still indexed after it is emptied; a read there would use an "
                 "uninitialised ciphertext")

    open(src, "w").write(txt)
    print("patch_park_encx_softmax: enc_X parked across the softmax, enc_X_v freed after its last read")


if __name__ == "__main__":
    main()
