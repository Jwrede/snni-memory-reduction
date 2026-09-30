#!/usr/bin/env python3
"""MOAI-GPU s2: release each bootstrap input after its own bootstrap_3 returns.

Basis: s1_complexroots_free VRAM peak (136,478,720 kB); VRAM attacked, host skipped. All four
bootstrap loops keep all 768 inputs until a swap(IN) after the loop; all four are patched (reserved
high-water set at bootstrap 3, monotone). Free (dead-value release). Gate T1s (modulus-chain trace),
byte for byte.
"""
import os
import re
import sys

# The four bootstrap loops, keyed by the input array each consumes; each is followed by
# `vector<PhantomCiphertext>().swap(<input>)`, which this lever moves earlier and per element.
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
        # Anchored on the exact call so a restructured tree fails loudly.
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

    # The marker the harness greps for and the build checks in the binary; placed at the first site only.
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
