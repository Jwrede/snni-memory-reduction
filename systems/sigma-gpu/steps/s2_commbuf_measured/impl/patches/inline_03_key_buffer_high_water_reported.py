# EXTRACTED, NOT AUTHORED. This is block 3 of
# `build_step_binary.sbatch`, lifted out verbatim by `tools/extract_inline_patches.py`.
#
# The script is unchanged and is still what produced the published binaries; this file exists
# so that one step's lever can be picked up on its own. The two are compared byte for byte by
# `--check`, so neither can drift from the other.
#
# Applied in the script as: python3 - "$G/experiments/sigma/sigma.cu" "$G/utils/sigma_comms.cpp" <<...
# Take it as: python3 inline_03_key_buffer_high_water_reported.py <the same target path>

import sys
sig, comms = sys.argv[1], sys.argv[2]

s = open(sig).read()
old = "    sigma->keySize = sigmaKeygen->keySize;"
if s.count(old) != 1:
    sys.exit("FAILED: the keySize handover matched %d times, expected 1" % s.count(old))
new = old + """
    // p_bufhighwater: what the key buffer actually holds, against what was reserved for it.
    fprintf(stderr, "SNNI_BUF|keybuf_used=%lu|keybuf_alloc=%lu\\n",
            (unsigned long)sigmaKeygen->keySize, (unsigned long)keyBufSz);
    fflush(stderr);"""
open(sig, "w").write(s.replace(old, new, 1))
print("  patched: key buffer high-water reported (experiments/sigma/sigma.cu)")

c = open(comms).read()
old2 = "    assert(memSz < commBufSize);"
if c.count(old2) != 1:
    sys.exit("FAILED: the commBufSize assert matched %d times, expected 1" % c.count(old2))
new2 = """    // p_bufhighwater: remember the largest request ever made of the comm buffer. Reported on
    // every new maximum rather than at exit, because this function has no exit hook and the last
    // line printed is the high-water.
    {
        static size_t snni_max_memsz = 0;
        if (memSz > snni_max_memsz) {
            snni_max_memsz = memSz;
            fprintf(stderr, "SNNI_BUF|commbuf_used=%lu|commbuf_alloc=%lu\\n",
                    (unsigned long)snni_max_memsz, (unsigned long)commBufSize);
            fflush(stderr);
        }
    }
""" + old2
open(comms, "w").write(c.replace(old2, new2, 1))
print("  patched: comm buffer high-water reported (utils/sigma_comms.cpp)")
