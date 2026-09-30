#!/usr/bin/env python3
"""MOAI-CPU m1_boot_out_move: the 4th bootstrap copies its output twice from a then-dead source (rtn2); make the second assignment std::move, redirecting the one diagnostic that reads rtn2[0]. FREE.

Usage: python3 patch_boot_out_move.py <path to the moai source tree>
"""
import os
import re
import sys

COPY = "            enc_ecd_x_copy[i*6+j] = rtn2[i*6+j];"
MOVED = """            // m1_boot_out_move: MOVE, do not copy. `rtn2` is not read again after this line
            // (the only later mention is the diagnostic below, redirected with it), so the stock
            // code leaves three 768-ciphertext arrays live at the peak where two suffice.
            enc_ecd_x_copy[i*6+j] = std::move(rtn2[i*6+j]);"""

DIAG = ('    cout <<"Modulus chain index after bootstrapping: "<< '
        'context.get_context_data(rtn2[0].parms_id())->chain_index()<<endl;')
DIAG_NEW = ('    // m1_boot_out_move: reads the ciphertext that now HOLDS the value, since rtn2[0]\n'
            '    // has been moved from. Same number as before.\n'
            '    cout <<"Modulus chain index after bootstrapping: "<< '
            'context.get_context_data(enc_ecd_x_copy[0].parms_id())->chain_index()<<endl;\n'
            '    cerr <<"LEVER|boot_out_move|moved_ciphertexts=768"<<endl;')


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], "include/test/test_full_scheme.hpp")
    if not os.path.exists(src):
        sys.exit("FAILED: no test_full_scheme.hpp under " + sys.argv[1])
    txt = open(src).read()

    if "LEVER|boot_out_move" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    if txt.count(COPY) != 1:
        sys.exit(f"FAILED: the second copy site matched {txt.count(COPY)} times, expected exactly 1")
    # The diagnostic text appears twice (after the 3rd and 4th bootstrap); only redirect the one
    # after the move site, anchored by position.
    if txt.count(DIAG) != 2:
        sys.exit(f"FAILED: the diagnostic matched {txt.count(DIAG)} times, expected exactly 2 "
                 "(after the third and fourth bootstrap)")
    diag_at = txt.find(DIAG, txt.index(COPY))
    if diag_at < 0:
        sys.exit("FAILED: no diagnostic after the move site")

    # Refuse if `rtn2` is read after the move site (comment bodies stripped first, since the block
    # comment below would otherwise read as live use); the move is safe only if the source is dead.
    tail = txt[txt.index(COPY) + len(COPY):]
    tail = re.sub(r"/\*.*?\*/", "", tail, flags=re.S)
    tail = re.sub(r"//[^\n]*", "", tail)
    uses = [m for m in re.finditer(r"\brtn2\b", tail)]
    # The diagnostic itself is the one permitted reader, and this patch redirects it.
    if len(uses) > 1:
        sys.exit(f"FAILED: `rtn2` is read {len(uses)} times after the move site; only the "
                 "diagnostic may remain, so moving here would consume something still live")

    # Replace the LATER diagnostic first, so the earlier one keeps its offsets and is untouched.
    txt = txt[:diag_at] + DIAG_NEW + txt[diag_at + len(DIAG):]
    txt = txt.replace(COPY, MOVED, 1)

    if "#include <utility>" not in txt:
        first = txt.index("#include")
        txt = txt[:first] + "#include <utility>   // m1_boot_out_move: std::move\n" + txt[first:]

    open(src, "w").write(txt)
    print("patch_boot_out_move: the second bootstrap-output copy is now a move, diagnostic redirected")


if __name__ == "__main__":
    main()
