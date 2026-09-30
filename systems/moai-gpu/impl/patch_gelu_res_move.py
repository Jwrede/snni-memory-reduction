#!/usr/bin/env python3
"""MOAI-GPU s3: GELU's accumulator takes x_n[1] by move instead of copy.

Basis: s2_boot_input_release VRAM decomposition (123,928,576 kB), largest object with a free fix:
6442.5 MB, gelu_other.cuh:179 (res = x_n[1]). x_n[1] is unread after line 179. Replaced by
`res = std::move(x_n[i])`. Free (less work; bit-identical value; move is real: ciphertext.h:66
defaulted move assignment on a cuda_auto_ptr payload). Gate T1s.
"""
import os
import re
import sys

ANCHOR = """    if (i == 1)
    {
      res = x_n[i];
    }
    else
    {
      evaluator.add_inplace(res, x_n[i], stream);
    }
"""

PATCH = """    if (i == 1)
    {
      // s3_gelu_res_move: MOVE the first term into the accumulator, do not copy it.
      // Every later iteration reads a different x_n[i], so x_n[1] has no reader after this
      // line and the stock code keeps two copies of it resident for the rest of the loop.
      // PhantomCiphertext has a defaulted move assignment (ciphertext.h:66) and its payload
      // is a cuda_auto_ptr, so this hands the device allocation over rather than duplicating it.
      res = std::move(x_n[i]);
      // Announced ONCE per process, not once per call. The previous step on this system printed
      // from inside a hot function and buried the level trace under thousands of identical lines.
      static bool snni_said = false;
      if (!snni_said) { snni_said = true; printf("LEVER|gelu_res_move|armed\\n"); }
    }
    else
    {
      evaluator.add_inplace(res, x_n[i], stream);
    }
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], "src/include/source/non_linear_func/gelu_other.cuh")
    if not os.path.exists(src):
        sys.exit("FAILED: no gelu_other.cuh under " + sys.argv[1])
    txt = open(src).read()

    if "LEVER|gelu_res_move" in txt:
        sys.exit("FAILED: this tree already carries the lever")
    # `res = x_n[i];` appears twice (live at 179, and inside a commented-out second gelu_v2 near
    # 399); the anchor is the whole uncommented if/else, which the //-prefixed dead copy cannot match.
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the accumulator if/else matched {txt.count(ANCHOR)} times, expected 1")

    # Refuse if x_n[1] is read after the assignment. Comments stripped first (this file has large
    # commented-out blocks including a second gelu_v2). Only a literal x_n[1] counts.
    at = txt.index(ANCHOR)
    tail = txt[at + len(ANCHOR):]
    tail = re.sub(r"/\*.*?\*/", "", tail, flags=re.S)
    tail = re.sub(r"//[^\n]*", "", tail)
    end = tail.find("PhantomCiphertext gelu_v2")      # stop at the next function, if any
    if end > 0:
        tail = tail[:end]
    uses = re.findall(r"x_n\s*\[\s*1\s*\]", tail)
    if uses:
        sys.exit(f"FAILED: x_n[1] is read {len(uses)} more time(s) after the assignment; moving "
                 "here would consume something still live")

    txt = txt.replace(ANCHOR, PATCH, 1)
    if "#include <utility>" not in txt:
        first = txt.index("#include")
        txt = txt[:first] + "#include <utility>   // s3_gelu_res_move: std::move\n" + txt[first:]

    open(src, "w").write(txt)
    print("patch_gelu_res_move: the GELU accumulator now moves its first term instead of copying it")


if __name__ == "__main__":
    main()
