# p_prm0: m0 with the recording pool

Job 46403378, 2026-08-25, `zen4` exclusive, two layers, T=16, rc=0, wall 38,747 s. Image
`moai-cpu-p_prm0.sif` from the published `m0_default` image via `impl/make_prm_def.py` (recorder is the
only difference).

## Instrument cost

```
m0 published    polled_peak  402,436,136 kB
p_prm0          polled_peak  402,401,204 kB      -0.0087%
```

- Largest deviation over 29 phase markers 0.531% (band 0.53 pp). Recorder table 16 MB (0.004%).
- Wall 38,747 against 38,283 s (+1.21%).

## Two tables, one freeze

Resolved by `impl/prm_resolve.sbatch` (job 46411436), same maps and `--program-root`
(`logs/run1/resolve.out`).

pmrec (who made the pool grow; 73 sites, 100% attributed):

```
227,851.1 MB   55%      373 items   Ct_pt_matrix_mul.hpp:35:ct_pt_matrix_mul_wo_pre
153,476.5 MB   37%      212 items   Batch_encode_encrypt.hpp:32:batch_input
  7,264.7 MB    2%       39 items   Ct_ct_matrix_mul.hpp:29:ct_ct_matrix_mul_colpacking
```

Recording pool (live at the same instant; 18 sites, 239.1 GB):

```
234,877.6 MB   98%   10,818 items   gelu_others.hpp:143:gelu_v2
  1,440.1 MB    1%    1,368 items   test_full_scheme.hpp:464 <- SEALContext::SEALContext
    672.3 MB           43 items     Polynomial.cpp:487:homomorphic_poly_evaluation
```

373 items of 611 MB are pool chunks; 10,818 items of 21.7 MB are ciphertexts at a mod-switched level.

## Retention

```
peak                402.3 GB
live                239.1 GB    59%
retained by pool    163.2 GB    41%
```

- Same fullscan over 169 mappings: 911.5 MB unaccounted for pmrec, 172,847.6 MB for the pool recorder.
- Of pmrec's 411.0 GB, 20.0 MB is off SEAL call paths (largest non-SEAL row 19.4 MB).
- Galois keys: live early (`../p_poolrec/`), returned to the pool by the peak; `m1`'s -21.36 GB (first
  ordering) works because it prevents pool growth.

## Limits

- Not row-comparable to the earlier fixed-index table (two frames inserted by the recording pool:
  `Ciphertext::operator=` against `encrypt_zero_symmetric`; `logs/run1/crosswalk.out`, job 46415873).
- With this table rule 4 names `gelu_v2`; the line was re-derived (pool first).
