# m1_rtn_release_early: valid measurement, superseded table

-3.88% (400,914,264 -> 385,346,328 kB), rc=0, wall 83,998 s, `bigsmp`. Off the line: its target was
chosen against a decomposition that no longer exists.

| | |
|---|---|
| object | `rtn`, the bootstrap output (768 ciphertexts); last reader the FFN matmul; stock code keeps it until the end of the layer |
| declaration | `object_attacked: seal::Ciphertext::operator=`, `cost_class: free` |
| marker | `LEVER|rtn_release_early|freed_ciphertexts=768`, once per layer |
| measured against the n=3 baseline | 402,960,964 -> 385,350,428 kB (-4.37%); wall 82,031 -> 83,998 s (+2.4%, n=1, free by mechanism) |

At `m0_default`'s peak (completed 2026-08-18) the targetable ranking was:

```
103,733.8 MB  25.3%  seal::util::encrypt_zero_symmetric      <- rank 1
 35,239.2 MB   8.6%  seal::Evaluator::mod_switch_scale_to_next
 32,013.5 MB   7.8%  Ct_pt_matrix_mul.hpp:87:ct_pt_matrix_mul_wo_pre_large   <- a PROGRAM line
 29,360.3 MB   7.2%  seal::Encryptor::encrypt_zero_internal
 28,533.3 MB   7.0%  seal::Evaluator::rotate_vector
 28,149.1 MB   6.9%  test_full_scheme.hpp:522:all_layer_test()               <- a PROGRAM line
```

- Build: "attacked 'seal::Ciphertext::operator=' but the largest object at the previous peak was
  'heap:seal::util::encrypt_zero_symmetric'". Rank 3 is a program line without a disposition.
- All six rows end in `<- seal::util::MemoryPoolHeadMT::get()`; the pool census measured 176.9 GiB
  (44.3% of a 400 GiB peak) as released memory held by the pool.
- `patch_rtn_release_early.py` predicted zero without `pool_new`; this binary has
  `rtn_release_early=1`, `MMProfNew=0` and moved the peak (block reuse, as MOAI-GPU's
  `s2_boot_input_release`).
- The declaration rested on run3's depth-5 table; the published median table (run2, depth 1) named only
  the allocator:

```
run1  depth 1   `MemoryPoolHeadMT::get()` at 99.3%          the allocator, not an object
run2  depth 1   the same
run3  depth 5   Ciphertext::operator=  228,647.9 MB  55.4%  <- program sites at last
                Ciphertext::resize     153,244.5 MB  37.1%
```

Peaks 402,690,024 / 402,965,064 / 403,306,200 kB (0.15%).
