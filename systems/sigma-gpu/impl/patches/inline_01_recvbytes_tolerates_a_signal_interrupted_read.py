# EXTRACTED, NOT AUTHORED. This is block 1 of
# `build_step_binary.sbatch`, lifted out verbatim by `tools/extract_inline_patches.py`.
#
# The script is unchanged and is still what produced the published binaries; this file exists
# so that one step's lever can be picked up on its own. The two are compared byte for byte by
# `--check`, so neither can drift from the other.
#
# Applied in the script as: python3 - "$G/utils/sigma_comms.cpp" <<...
# Take it as: python3 inline_01_recvbytes_tolerates_a_signal_interrupted_read.py <the same target path>

import sys
p = sys.argv[1]
s = open(p).read()
old = """        assert(numRead == toRead);
        bytesRead += numRead;"""
if s.count(old) != 1:
    sys.exit("FAILED: the recvBytes assert matched %d times, expected 1" % s.count(old))
new = """        // SNNI setup: a short return is a signal-interrupted MSG_WAITALL, not an error. The
        // loop this sits in already handles it -- `bytesRead` advances and the next iteration
        // asks for the remainder -- so the assert was the only thing turning it into a crash.
        // A genuine close returns 0 and would spin here, so that case is still fatal.
        assert(numRead > 0 && "peer closed mid-transfer");
        bytesRead += numRead;"""
open(p, "w").write(s.replace(old, new))
print("  patched: recvBytes tolerates a signal-interrupted read (setup)")
