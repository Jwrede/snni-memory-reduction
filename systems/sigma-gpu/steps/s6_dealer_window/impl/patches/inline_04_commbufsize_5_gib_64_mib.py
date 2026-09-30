# EXTRACTED, NOT AUTHORED. This is block 4 of
# `build_step_binary.sbatch`, lifted out verbatim by `tools/extract_inline_patches.py`.
#
# The script is unchanged and is still what produced the published binaries; this file exists
# so that one step's lever can be picked up on its own. The two are compared byte for byte by
# `--check`, so neither can drift from the other.
#
# Applied in the script as: python3 - "$G/utils/sigma_comms.h" <<...
# Take it as: python3 inline_04_commbufsize_5_gib_64_mib.py <the same target path>

import sys
p = sys.argv[1]
s = open(p).read()
old = "    size_t commBufSize = 5 * OneGB;"
if s.count(old) != 1:
    sys.exit("FAILED: commBufSize matched %d times, expected exactly 1" % s.count(old))
new = """    // s2_commbuf_measured: 5 GiB stood ready for a message measured at 384 KiB.
    //
    // `p_bufhighwater` recorded the largest request this buffer ever sees (job 46106974): 393,216
    // bytes against 5,368,709,120 reserved, a factor of 13,653, and the class holds TWO of them.
    // The buffer carries ONE message at a time (every use passes `memSz`; gpu_comms.h:215-251),
    // so the reserve serves no accumulation.
    //
    // 64 MiB is 170x the largest message observed. The margin is wide on purpose: the measurement
    // is one run of one pinned workload. Undersizing cannot corrupt a result, because
    // `getMemSz` asserts `memSz < commBufSize` and aborts; oversizing only keeps the memory this
    // step removes.
    size_t commBufSize = 64ULL * 1024 * 1024;"""
open(p, "w").write(s.replace(old, new, 1))
print("  patched: commBufSize 5 GiB -> 64 MiB (utils/sigma_comms.h)")
