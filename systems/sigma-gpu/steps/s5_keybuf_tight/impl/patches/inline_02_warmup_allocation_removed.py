# EXTRACTED, NOT AUTHORED. This is block 2 of
# `build_step_binary.sbatch`, lifted out verbatim by `tools/extract_inline_patches.py`.
#
# The script is unchanged and is still what produced the published binaries; this file exists
# so that one step's lever can be picked up on its own. The two are compared byte for byte by
# `--check`, so neither can drift from the other.
#
# Applied in the script as: python3 - "$G/utils/gpu_mem.cu" <<...
# Take it as: python3 inline_02_warmup_allocation_removed.py <the same target path>

import sys
p = sys.argv[1]
s = open(p).read()
old = """    uint64_t *d_dummy_ptr;
    uint64_t bytes = 40 * (1ULL << 30);
    checkCudaErrors(cudaMallocAsync(&d_dummy_ptr, bytes, 0));
    checkCudaErrors(cudaFreeAsync(d_dummy_ptr, 0));
"""
if s.count(old) != 1:
    sys.exit("FAILED: the warmup block matched %d times, expected exactly 1" % s.count(old))
new = """    // s1_nowarmup: the 40 GiB warmup allocation is removed.
    //
    // It allocated 40 GiB and freed it on the next line. Nothing wrote to it, nothing read it, and
    // the pointer never left this function -- its only purpose was to force the pool to RESERVE
    // 40 GiB up front. Because the release threshold above is UINT64_MAX the pool never returned
    // those pages, so the free was not a free and the device held 40 GiB for the whole run.
    //
    // This campaign's own device table names it as 100% of the VRAM peak. Removing it cannot change
    // a value, because a buffer written by nothing and read by nothing carries none. The pool grows
    // on demand instead, which is the runtime cost this step is measured on.
"""
open(p, "w").write(s.replace(old, new))
print("  patched: warmup allocation removed (utils/gpu_mem.cu)")
