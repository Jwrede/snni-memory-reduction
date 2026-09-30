#!/usr/bin/env python3
"""SIGMA-GPU diagnostic: prints the size of a failing device allocation (p_seq197's crash reports
none). Wraps the failing gpuMalloc, prints bytes/free/total and a call counter, then passes the same
error to checkCudaErrors. No behaviour change.

    patch_alloc_diag.py <sigma source tree>

    SNNI_ALLOC|FAILED|bytes=..|free=..|total=..|call=..
"""
import os
import sys

SRC = "utils/gpu_mem.cu"

OLD = """extern "C" uint8_t *gpuMalloc(size_t size_in_bytes)
{
    uint8_t *d_a;
    checkCudaErrors(cudaMallocAsync(&d_a, size_in_bytes, 0));
    return d_a;
}"""

NEW = """extern "C" uint8_t *gpuMalloc(size_t size_in_bytes)
{
    uint8_t *d_a;
    // SNNI diagnostic: on failure, say HOW BIG the request was. `checkCudaErrors` reports the file,
    // the line and the expression but not the size, and the size is the only thing that identifies
    // which computation produced an impossible request. Behaviour is unchanged: the same error is
    // handed to the same macro immediately after, so the abort and the exit code are as before.
    static unsigned long snni_alloc_calls = 0;
    snni_alloc_calls++;
    cudaError_t snni_e = cudaMallocAsync(&d_a, size_in_bytes, 0);
    if (snni_e != cudaSuccess) {
        size_t snni_free = 0, snni_total = 0;
        cudaMemGetInfo(&snni_free, &snni_total);
        fprintf(stderr, "SNNI_ALLOC|FAILED|bytes=%zu|free=%zu|total=%zu|call=%lu\\n",
                size_in_bytes, snni_free, snni_total, snni_alloc_calls);
        fflush(stderr);
    }
    checkCudaErrors(snni_e);
    return d_a;
}"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.join(sys.argv[1], SRC)
    if not os.path.exists(src):
        sys.exit("FAILED: no " + SRC + " under " + sys.argv[1])
    txt = open(src).read()

    if "SNNI_ALLOC|" in txt:
        sys.exit("FAILED: this tree already carries the diagnostic")
    if txt.count(OLD) != 1:
        sys.exit(f"FAILED: gpuMalloc's body matched {txt.count(OLD)} times, expected exactly 1. "
                 f"Another patch reshaped it; re-derive rather than loosening the anchor.")

    open(src, "w").write(txt.replace(OLD, NEW, 1))
    if "SNNI_ALLOC|FAILED" not in open(src).read():
        sys.exit("FAILED: applied the patch but its marker is absent")
    print("patch_alloc_diag: failing device allocations now report their size (" + SRC + ")")


if __name__ == "__main__":
    main()
