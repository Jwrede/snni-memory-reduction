#!/usr/bin/env python3
"""MOAI-GPU s2: release each bootstrap input after its own bootstrap_3 returns.

Basis: s1_complexroots_free VRAM peak (136,478,720 kB); VRAM attacked, host skipped (levers.env).
Markers (vram_used_MiB):
    keys_created          41,245 MiB
    before_attention      95,039
    after_bootstrap1     108,835      (+13,792)
    after_bootstrap2     115,843      (+ 7,008)
    after_bootstrap3     133,955      (+18,112)   <- reserved 133,280 MiB, the published peak
    after_bootstrap4     133,955
Maximum set at bootstrap 3 (monotone high-water): all four bootstrap loops patched.
Loop shape:
    vector<PhantomCiphertext> OUT(768);
    for (i = 0..127) for (j = 0..5)
        bootstrapper.bootstrap_3(OUT[i*6+j], IN[i*6+j]);
    ...
    vector<PhantomCiphertext>().swap(IN);          // the whole input, AFTER the loop
IN[i*6+j] dead after its bootstrap_3; released per element. Free (dead-value release; order unchanged).
Source checks: bootstrap_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher) does not retain
its input; PhantomCiphertext has a default constructor and defaulted move assignment
(include/ciphertext.h:59,67) over a cuda_auto_ptr (assignment releases the buffer).
Skip: PhantomSecretKey::encrypt_zero_symmetric (41.32 GiB, 1121 x 37.7 MB) = key bank (gen_publickey,
gen_relinkey, gal_keys_boot, test_single_layer.cuh:437-440); one Galois set (matmul set commented
out), so MOAI-CPU's phase split does not apply; regeneration is paid.
Falsifier: gate T1s (modulus-chain trace) must be byte-identical.

Usage: python3 patch_boot_input_release.py <path to prepared source tree>
"""
import os
import re
import sys

# The four bootstrap loops, keyed by the input array each consumes. Every one of them is followed a
# few lines later by `vector<PhantomCiphertext>().swap(<input>)`, which is the program's own
# statement that the array is dead at that point; this lever only moves that statement earlier and
# per element.
SITES = [
    ("rtn",  "att_selfoutput"),          # bootstrap1
    ("rtn",  "layernorm_selfoutput"),    # bootstrap2
    ("rtn2", "final_output"),            # bootstrap3, the one that sets the peak
    ("rtn2", "layernorm_finaloutput"),   # bootstrap4
]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], "src/include/test/test_single_layer.cuh")
    if not os.path.exists(src):
        sys.exit("FAILED: no test_single_layer.cuh under " + sys.argv[1])
    txt = open(src).read()

    if "LEVER|boot_input_release" in txt:
        sys.exit("FAILED: this tree already carries the lever")

    for out, inp in SITES:
        call = f"bootstrapper.bootstrap_3({out}[i*6+j],{inp}[i*6+j]);"
        # Anchored on the exact call so a tree whose loops were restructured fails loudly rather
        # than being patched somewhere that looks similar.
        m = re.search(r"([ \t]*)" + re.escape(call), txt)
        if not m:
            sys.exit(f"FAILED: call site not found: {call}")
        if txt.count(call) != 1:
            sys.exit(f"FAILED: call site is not unique ({txt.count(call)}x): {call}")
        ind = m.group(1)
        repl = (f"{ind}{call}\n"
                f"{ind}// s2 boot_input_release: this input is dead the instant its own bootstrap\n"
                f"{ind}// returns; the stock code holds all 768 until after the loop.\n"
                f"{ind}{inp}[i*6+j] = PhantomCiphertext();")
        txt = txt.replace(m.group(0), repl, 1)

    # The announcement the harness greps for, and the marker the build checks in the binary. Placed
    # at the first site rather than at every one, so it says "the lever is compiled in" once.
    first = f"{SITES[0][1]}[i*6+j] = PhantomCiphertext();"
    txt = txt.replace(
        first,
        first + '\n            if (i == 0 && j == 0) '
                'std::cerr << "LEVER|boot_input_release|sites=4" << std::endl;',
        1)

    open(src, "w").write(txt)
    print(f"patch_boot_input_release: {len(SITES)} bootstrap loops patched in {src}")


if __name__ == "__main__":
    main()
