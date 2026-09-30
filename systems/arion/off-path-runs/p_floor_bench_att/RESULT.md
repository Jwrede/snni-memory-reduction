# p_floor_bench_att: ARION QK^T instant

| | |
|---|---|
| job | 47044979, 2026-09-29, PALMA `normal`, r02n07, T=1, GOMAXPROCS 1 |
| program | `impl/floor_bench_att/`, built in `arion-a4_activation_input_release_goheap.sif` against Lattigo v6.1.1 and `configs/config.go` SetHEParams |
| method | the per-head QK^T working set as one set, every page touched (keys generated, ciphertexts encrypted), VmRSS after GC and `FreeOSMemory` per stage, context subtracted |
| analogue | MOAI-CPU `off-path-runs/p_floor_bench_att` |

```
component                       measured kB   derived kB
30 attention Galois keys          2,348,156    2,334,720   +0.6%
relinearisation key                  30,784       77,824   (heap reuse after the key generation)
Q, K, rotated Q, rotated K, V     4,602,808    4,587,520   +0.3%   (5 x 64 ct at level 13)
128 score diagonals               1,667,868    1,703,936   -2.1%   (level 12)
instant - context                 8,649,616    8,704,000   -0.62%
```

Integrated: 8.86 GB against 8.91 GB derived (decimal). Per-component deltas partly offset (freed
key-generation buffers are reused).
