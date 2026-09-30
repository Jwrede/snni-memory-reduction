# p_poolrec: recording SEAL pool, smoke test

Job 46403188, 2026-08-25, `express`, one layer, T=16, killed at 12 min (rc=137, wall 634 s). Image
`moai-cpu-p_prm0.sif` (`m0_default` plus the recording pool).

```
peak                82,873,792 kB     in the band of the depth probes (82.5 to 84.6 GB)
both tables         sample_ts_ms=1787565579174, sample_rss_kb=82119676   IDENTICAL
pmrec               83.7 GB over 25 rows
pool recorder       83.6 GB over 15 rows
```

- Both tables in one SIGSTOP, resident bytes counted by `pmsample` over each item's range.
- Pool table early: 80.2 GB in 2,124 items of 37.75 MB (a ciphertext is 37.7 MB, a Galois key ~36 of
  them): the key material, live. At `m0`'s peak two such items remain: keys returned to the pool.
- Counters: 870,476,903 inserts, 0 drops; a failed `forget` would keep items live, not lose them.
- Composition at the peak: `../p_prm0/`.
