#!/usr/bin/env python3
"""MOAI-GPU s3: mod-switch copies a whole ciphertext to protect one limb (REFUTED).

Basis: s2_boot_input_release VRAM decomposition (123,928,576 kB), mod_switch_scale_to_next 31.52 GiB.
Each mod-switch copies its input because divide_and_round_q_last_ntt's in-place inverse NTT destroys
it; assumed to write only the last limb. Saves/restores the last limb instead. Free (two limb
transfers vs twenty). Gate T1s. Refuted: segfault after 7 of 12 level transitions (job 46120481);
off-path.
"""
import os
import sys

ANCHOR = """    auto encrypted_copy = make_cuda_auto_ptr<uint64_t>(encrypted_size * coeff_mod_size * poly_degree, stream);
    cudaMemcpyAsync(encrypted_copy.get(), encrypted.data(),
                    encrypted_size * coeff_mod_size * poly_degree * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);
"""

PATCH = """    // s3_modswitch_lastlimb: save ONE limb, not the whole ciphertext.
    //
    // The stock code copies encrypted_size * coeff_mod_size * poly_degree words to protect the
    // input, because divide_and_round_q_last_ntt begins with an in-place inverse NTT. That call
    // writes ONE polynomial, at modulus index base_q_size - 1: only the LAST limb. Everything else
    // it reads and leaves alone. So the last limb is saved here, the stock computation runs
    // against the ciphertext itself, and the limb is put back afterwards. `destination` sees the
    // same bytes and `encrypted` ends bit for bit as it began.
    //
    // At the levels this system peaks on, coeff_mod_size is about twenty, so this is two
    // limb-sized transfers where there were twenty.
    auto snni_last_limb = make_cuda_auto_ptr<uint64_t>(encrypted_size * poly_degree, stream);
    auto *snni_src = const_cast<uint64_t *>(encrypted.data());
    for (size_t snni_i = 0; snni_i < encrypted_size; snni_i++) {
        cudaMemcpyAsync(snni_last_limb.get() + snni_i * poly_degree,
                        snni_src + snni_i * coeff_mod_size * poly_degree
                                 + (coeff_mod_size - 1) * poly_degree,
                        poly_degree * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    }
    printf("LEVER|modswitch_lastlimb|limbs=%zu\\n", coeff_mod_size);
"""

RESTORE_ANCHOR = """    switch (next_parms.scheme()) {
        case scheme_type::bfv:
            rns_tool.divide_and_round_q_last(encrypted_copy.get(), encrypted_size, destination.data(), stream);
            break;"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], "thirdparty/phantom-fhe/src/evaluate.cu")
    if not os.path.exists(src):
        sys.exit("FAILED: no phantom-fhe/src/evaluate.cu under " + sys.argv[1])
    txt = open(src).read()

    if "LEVER|modswitch_lastlimb" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the defensive copy matched {txt.count(ANCHOR)} times, expected exactly 1")
    if txt.count(RESTORE_ANCHOR) != 1:
        sys.exit(f"FAILED: the scheme switch matched {txt.count(RESTORE_ANCHOR)} times, expected 1")

    # The stock code reads `encrypted_copy` in three branches, all of which must now read the
    # ciphertext itself (expected 4 uses: memcpy dest + three scheme branches).
    n_uses = txt.count("encrypted_copy.get()")
    if n_uses != 4:
        sys.exit(f"FAILED: `encrypted_copy.get()` appears {n_uses} times, expected 4: once as the "
                 "memcpy destination and once in each of the three scheme branches. This patch was "
                 "written against that shape")

    # Locate the switch before any renaming: the anchor text contains the very name about to change.
    txt = txt.replace(ANCHOR, PATCH, 1)
    # The anchor carried the memcpy, so exactly the three READERS are left to redirect.
    left = txt.count("encrypted_copy.get()")
    if left != 3:
        sys.exit(f"FAILED: {left} readers left after removing the copy, expected 3")
    txt = txt.replace("encrypted_copy.get()", "snni_src")
    if "encrypted_copy" in txt:
        sys.exit("FAILED: `encrypted_copy` survives somewhere; a reader was missed and would not "
                 "compile, but a surviving name could silently keep the old behaviour")

    # Restore after whichever branch ran: placed immediately after the switch's closing brace.
    at = txt.index("switch (next_parms.scheme())")
    depth = 0
    i = txt.index("{", at)
    while True:
        if txt[i] == "{":
            depth += 1
        elif txt[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    end = txt.index("\n", i) + 1
    restore = """
    // s3_modswitch_lastlimb: put the limb back, so the caller's ciphertext is unchanged.
    for (size_t snni_i = 0; snni_i < encrypted_size; snni_i++) {
        cudaMemcpyAsync(snni_src + snni_i * coeff_mod_size * poly_degree
                                 + (coeff_mod_size - 1) * poly_degree,
                        snni_last_limb.get() + snni_i * poly_degree,
                        poly_degree * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    }
"""
    txt = txt[:end] + restore + txt[end:]

    open(src, "w").write(txt)
    print("patch_modswitch_lastlimb: the mod-switch now guards one limb instead of the whole ciphertext")


if __name__ == "__main__":
    main()
