#!/usr/bin/env python3
"""Applies s2_modraise_srclimb to a prepared moai-gpu source tree (patch_modraise.py <src-tree>).

modraise_inplace's full-level copy replaced by a buffer of the N real values plus N zeros the kernels
read (upstream's zeros reproduced byte for byte).
"""
import glob
import sys

root = sys.argv[1].rstrip("/")

# Located by glob, not hardcoded; exactly one match required (two would risk patching an uncompiled copy).
hits = glob.glob(root + "/**/include/source/bootstrapping/Bootstrapper.cu", recursive=True)
if len(hits) != 1:
    sys.exit("FAILED: %d copies of Bootstrapper.cu under %s, expected exactly 1: %s"
             % (len(hits), root, hits))
p = hits[0]
print("  patching %s" % p)
s = open(p).read()

old = """  auto ciphertext_size = cipher.size();
  PhantomCiphertext cipher_copy = cipher;

  uint64_t gridDimGlb = N / blockDimGlb.x;
  for (size_t poly_idx = 0; poly_idx < ciphertext_size; poly_idx++)
  {
    const auto rns_poly_src = cipher_copy.data() + poly_idx * rns_coeff_count;
"""
if s.count(old) != 1:
    sys.exit("FAILED: the modraise copy block matched %d times, expected 1" % s.count(old))

new = """  auto ciphertext_size = cipher.size();

  // s2_modraise_srclimb: copy only what the kernels read, not the whole full-level ciphertext.
  //
  // The original took `PhantomCiphertext cipher_copy = cipher` AFTER the resize, i.e.
  // `2 * mod_count * N` u64 = 37.7 MB per call at this system's parameters, and read from it at
  // stride `mod_count * N`. What those reads actually touch is N values per polynomial:
  //
  //   poly 0 at offset 0            polynomial 0's limb 0, the real data
  //   poly 1 at offset mod_count*N  inside the region `resize` zeroed, because the resize copies
  //                                 the old 2N values CONTIGUOUSLY to the front while the layout
  //                                 is polynomial-major (ciphertext.h:279)
  //
  // Both are reproduced here byte for byte, zeros included. That the second polynomial is
  // mod-raised from zeros is upstream's behaviour; a reduction step may not change a value, so it
  // is preserved rather than corrected.
  // The CUDA calls are unwrapped, matching this file's own style: `checkCudaErrors` lives in
  // `cuda_wrapper.cuh`, which `Bootstrapper.cu` does not include, and the first attempt at this
  // patch failed to compile for exactly that (job 45885137, 63 s). Adding the include would put a
  // header into a translation unit for a reduction step, which is more change than the step needs.
  auto src_limb = phantom::util::make_cuda_auto_ptr<uint64_t>(ciphertext_size * N, stream);
  cudaMemsetAsync(src_limb.get(), 0, ciphertext_size * N * sizeof(uint64_t), stream);
  cudaMemcpyAsync(src_limb.get(), cipher.data(), N * sizeof(uint64_t),
                  cudaMemcpyDeviceToDevice, stream);

  uint64_t gridDimGlb = N / blockDimGlb.x;
  for (size_t poly_idx = 0; poly_idx < ciphertext_size; poly_idx++)
  {
    const auto rns_poly_src = src_limb.get() + poly_idx * N;
"""
s = s.replace(old, new)
open(p, "w").write(s)
print("  patched: modraise_inplace copies one limb per polynomial (s2_modraise_srclimb)")

# Read back from the file, so the check is on what was written.
t = open(p).read()
for must in ("src_limb.get() + poly_idx * N", "make_cuda_auto_ptr<uint64_t>(ciphertext_size * N"):
    if must not in t:
        sys.exit("FAILED: %r is not in the patched file" % must)
if "PhantomCiphertext cipher_copy = cipher;" in t:
    sys.exit("FAILED: the full-level copy is still there")
print("  verified against the file")
