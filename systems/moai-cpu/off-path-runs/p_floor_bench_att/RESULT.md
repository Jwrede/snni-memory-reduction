# p_floor_bench_att: MOAI QK^T instant, measured

Floor micro-benchmark at T=1 of one head's `QK^T` working set, VmRSS from `/proc/self/status`
(complements `p_floor_bench/`, the bootstrap instant).

| | |
|---|---|
| patcher | `impl/p_floor_bench_att.py` (patches `test_full_scheme.hpp`): body inserted right after `create_relin_keys`, before the 30-key and 47-key banks; local encoder/encryptor/evaluator; baseline after the public key; 14 rotation keys; head blocks built by copying a right-sized template per level (encrypt at the top level, mod-switch down, copy to a tight allocation); returns before inference |
| run | `SNNI_FLOOR_ATT=1 OMP_NUM_THREADS=1` |
| build | job 47030624 (`normal`), `moai-cpu-p_floor_bench_att.sif` from `moai-cpu-m0_default.sif`, BUILD_OK |
| measure | job 47030625 (zen4 r19n28, T=1), `FLOOR_ATT_SUMMARY`, wall ~2 min |

```
FLOOR_ATT|ctx_pub        vmrss_kb=3,746,472    context + public key (baseline, before relin)
FLOOR_ATT|att_context    vmrss_kb=5,133,204    + relin key (delta 1,386,732 kB = 1.42 GB)
FLOOR_ATT|att_after_keys vmrss_kb=23,202,996    + 14 rotation keys (delta 18,069,792 kB)
FLOOR_ATT|att_instant    vmrss_kb=28,581,272    + one head's Q, K, copy, scores, V (delta 5,378,276 kB)
FLOOR_ATT_SUMMARY|ctx_pub_kb=3,746,472|keys14_kb=18,069,792|blocks_kb=5,378,276
                 |instant_kb=28,581,272|target_kb=24,834,800
```

## Against the derivation of the built set (14 keys)

```
term            derived kB     measured kB    note
14 keys         18,063,360     18,069,792     0.04% high; the dominant term, exact
relin           1,290,240      1,386,732      + the small public key above it in the delta
blocks          4,653,056      5,378,276      + ~0.85 GB SEAL pool retention from the T=1
                                               template construction (the same retention the
                                               line's pool step addresses; a T=1 bench has no
                                               threshold pool)
-------------------------------------------------------------------------------------------
target          24,006,656     24,834,800     +3.4%, the excess being pool retention
```

- The measurement lies above the derivation (retention), the key term exact.

## The attention's distinct keys

The bench builds 14 keys, one per requested rotation step. At 32,768 slots +16,384 and -16,384 map to
the same Galois element: the attention needs 13 distinct keys, and its instant is 22,716,416 kB
(23.3 GB), below the chunk-128 bootstrap instant that sets the target (25,267,200 kB, 25.9 GB,
MANIFEST.md). The comparison above holds for the 14-key set the bench built (derived 24,006,656 kB).
