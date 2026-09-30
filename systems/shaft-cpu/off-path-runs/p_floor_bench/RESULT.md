# p_floor_bench: SHAFT BERT target, measured

Target micro-benchmark of the `shaft-cpu` target (MANIFEST, thesis Appendix B.2): the embedding
lookup at the partition of `s7_embed_chunk`, 16 vocabulary slices of 1,813 rows. Each term is built
with CrypTen's own objects and its resident footprint read from `/proc/self/status` (VmRSS).

| | |
|---|---|
| script | `shaft_floor_bench.py` |
| image | docker `shaft-cpu-s0_default` (`sha256:13342378...`), torch 2.0.1+cpu, crypten 1.0.0, provider `TrustedFirstParty` |
| machine | local VPS, 4 cores, `OMP_NUM_THREADS=4`, container limit 3 GB; one process, `WORLD_SIZE=1` |
| method | shares via `crypten.cryptensor`, the slice triple via `provider.generate_additive_triple((128,1813),(1813,768),"matmul")`; every page touched; after each term the plaintext source is dropped, `gc.collect()` and `malloc_trim(0)` run, so a delta counts live shares and not glibc retention |
| runs | 3 (`run_local_1.out` to `run_local_3.out`), ~1 min each |

```
term             derived B      tensor B       RSS delta (KiB), runs 1 / 2 / 3
table share      178,151,424    178,151,424    178,008 / 176,024 / 178,048
one-hot          29,691,904     29,691,904      30,376 /  30,508 /  30,508
slice triple     13,782,016     13,782,016      14,360 /  14,364 /  14,360
output + sum      1,572,864      1,572,864       1,536 /   1,536 /   1,540
target          223,198,208    223,198,208     224,280 / 222,432 / 224,456 KiB
                                               = 229.7 / 227.8 / 229.8 MB, +2.9 / +2.1 / +3.0 %
```

- Every term's tensors have exactly the derived size (int64 shares of the derived shapes).
- Resident lies 2.1 to 3.0 % above the derivation; the excess sits in the first share built (table,
  +2 to +4 MB, CrypTen and torch state on first use, as in `shaft-cpu-vit/off-path-runs/p_floor_bench`)
  and in the one-hot and triple terms (+0.9 to +1.5 MB each). The output term is exact.
- Like the other target micro-benchmarks it confirms the sizing of the terms; it does not test
  minimality, and it does not run the slice matmul itself.
