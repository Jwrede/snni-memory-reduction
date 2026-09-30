# m0_peak_table_preview: m0's table from the peak artefacts

Not a publication; the published table is from `derive.sbatch` (job 46371028). Same computation run
early on the poller's peak artefacts, separate output path.

```
peak.smaps   180,887 B   resident mappings at the peak
pm.maps       16,650 B   the maps snapshot the addresses resolve against
resident.txt  13,741 B   the pagemap aggregation, 312 lines
PEAK              24 B   1787420084853 402063680
```

Written at 19:34 CEST when `m0` passed `after_layernorm2`; `sample_rss_kb` 402,021,176 against `PEAK`
402,063,680 (0.011%).

Campaign resolver, job 46380301, 36 s, resident bytes:

```
resident     allocated  touched count  site
103,733.8 M  108,527.6M   96%    55  seal::util::encrypt_zero_symmetric          RANK 1, 25.2%
 33,023.8 M   34,672.2M   95%    42  seal::Evaluator::mod_switch_scale_to_next
 32,002.9 M   33,638.3M   95%    52  Ct_pt_matrix_mul.hpp:87:ct_pt_matrix_mul_wo_pre_large
 31,845.5 M   32,191.3M   99%    47  evaluator.h:1232:seal::Evaluator::rotate_vector
 29,360.3 M   29,360.1M  100%    35  seal::Encryptor::encrypt_zero_internal
 28,149.1 M   28,149.0M  100%    10  test_full_scheme.hpp:529:all_layer_test()
 18,815.7 M   18,815.6M  100%     9  Bootstrapper.cpp:2092:rotated_bsgs_linear_transform
 15,667.1 M   16,357.8M   96%    30  layernorm.hpp:320:layernorm
 14,195.8 M   14,195.6M  100%    33  Ct_ct_matrix_mul.hpp:28:ct_ct_matrix_mul_colpacking
 13,251.7 M   14,695.3M   90%    81  seal::Evaluator::ckks_multiply
 11,939.3 M   12,095.3M   99%    21  single_att_block.hpp:73:single_att_block

134 call sites, 410,968.7 MB attributed
639.4 MB of 411,669.7 MB covered by no recorded allocation at all, 0.16%
```

- Rank 1 `encrypt_zero_symmetric` (25.2%) is what `m1_minimal_att_keys` (first ordering) declared.
- `test_full_scheme.hpp:529`: copy loop of `enc_ecd_x_copy`, 28,149.1 MB, ten allocations, 100%
  touched.
- Against the depth-6 table on `bigsmp` (`m1_rtn_release_early` image): `encrypt_zero_symmetric`
  103,733.8 against 103,733.8 MB; `test_full_scheme.hpp:52x` 28,149.1 against 28,149.1 MB.
- First quoted from `peek_live.py` (allocated bytes); the agreement above is resident against resident.
