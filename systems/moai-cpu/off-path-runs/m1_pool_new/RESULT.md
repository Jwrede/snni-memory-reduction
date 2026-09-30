# m1_pool_new: SEAL MMProfNew as the first step (retired paid-first line)

n=1, gate `GATE_NONZERO 7680/7680`, rc=0.

```
m0_default    402,961,988 kB   wall  82,031 s   W 5.14
m1_pool_new   307,224,436      wall 100,351 s   W 3.92   -23.7%, factor 1.311x, +22.3% wall -> PAID
```

- Paid and first; the free `m1_rtn_release_early` (-4.37%) existed, so rule 6 put it first and
  `pool_new` was to be re-measured as `m2_pool_new`. That run was cancelled (user's decision) while the
  baseline was re-measured at a deeper recorder (job 46196739).
- BumbleBee's `x4_seal_pool_new` cost +12.7%; here +22.3%. Pool census predicted ~222 GiB, run reached
  293: with `MMProfNew` layer 1 cannot reuse layer 0's free list:

```
248.3 -> 256.6 -> 272.5 -> 290.0 -> 293.0 GiB
```

Table at depth 5, (caller, allocating function):

```
150.34 GiB  51.5%  seal::Ciphertext::resize     <- MemoryPoolHeadMT::MemoryPoolHeadMT
 89.57 GiB  30.7%  seal::Ciphertext::operator=  <- MemoryPoolHeadMT::MemoryPoolHeadMT
  2.67 GiB   0.9%  seal::util::CreateNTTTables  <- MemoryPoolHeadMT::get()
 24.92 GiB   8.5%  anon:[heap]                  located, never a target
 22.61 GiB   7.7%  anon:mmap-rest(831)          located, never a target
```

`patch_rtn_release_early.py` claimed its release only reaches the OS under `pool_new`;
`m1_rtn_release_early` without it (`MMProfNew=0`) moved the peak 4.37%: a release lowers the
high-water when a later same-shape allocation reuses the block (as MOAI-GPU). The published line's
pool step is `m1_pool_threshold`.
