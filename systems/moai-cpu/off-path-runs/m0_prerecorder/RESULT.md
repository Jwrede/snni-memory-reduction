# m0_prerecorder: baseline before the pool recorder

Job 46196739 generation, 2026-08-20, `zen4` exclusive, two layers, T=16, rc=0. Was `m0_default` until
2026-08-25.

```
peak_host   402,432,036 kB   W 5.13   (polled 402,436,136)
```

- Peak stands: the replacement reads 402,401,204 kB (-0.0087%), gate bit-identical over 7,680 values.
- `pmrec` alone: a destroyed `Ciphertext` returns to SEAL's pool, never to `free()`, so rows name the
  first requester of a chunk. Rank 1 `encrypt_zero_symmetric` (103,733.8 MB, 25.2%) aggregates a path;
  key material cannot appear as such; `gelu_v2` does not appear.
- Replacement (`../../steps/m0_default/`, run `p_prm0`): recording SEAL pool; ranks
  `gelu_others.hpp:143:gelu_v2` first (`tools/attrib_seal_pool/README.md`, `../p_prm0/RESULT.md`).
- `m1_minimal_att_keys` (first ordering) declared against this table: -21.36 GB measured; source
  derivation: 17 of 31 attention Galois keys never requested (~24.1 GB).
- Rows are not comparable to the replacement's row by row (the recording pool inserts two frames);
  same raw table resolved both ways: `../p_prm0/logs/run1/crosswalk.out`.
