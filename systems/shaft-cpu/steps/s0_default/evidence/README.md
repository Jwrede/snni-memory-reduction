# s0_default: object identity behind `heap:libc10.so+0x4a675`

| | |
|---|---|
| recorder row | `heap:libc10.so+0x4a675`, 2556484 kB, 505 allocations, 71% of the peak |
| meaning | return address of `c10::alloc_cpu` (torch CPU allocator, `posix_memalign`); every torch CPU tensor maps to it |
| deeper frames | `at::empty`, then CrypTen Python code; the identity exists only at the Python level |
| identity source | `impl/levers/probe_census_cpu.py` (diagnostic, never in a measured run) |

## Census run

| | |
|---|---|
| run | `results/p_census/run1`, job 45509249 |
| image | `f35e71f6d21c683f46f1872c2cca3276aa8fdd3813e117aff97fbded42a3abfe` |
| workload | 12 layers, seq 128, `SHAFT_LEVERS=probe_census_cpu` |
| party 0 pid | 438870 |
| method | at each new RSS maximum: walk `gc.get_objects()` for `torch.Tensor`, dedupe by storage pointer, group by `(shape, dtype)` |

The peak value comes from the measured run; the census supplies only identity.

## Cross-check against the measured run's full scan

Large tensors sit alone in page-aligned `mmap` mappings:

| census claim | bytes | mapping in the measured run |
|---|---:|---:|
| `int64 (28996, 768)`, word-embedding secret share | 178151424 | 178155520 |
| `float32 (28996, 768)`, plaintext word-embedding table | 89075712 | 89079808 |

Each mapping is the tensor plus one page. Totals: census 2589.5 MB of live tensors, recorder
2972.1 MB at the same site; the 382.6 MB gap is C++-internal temporaries without a Python reference.

## Peak location (`../logs/run1`, party 0 markers)

```
 -0.30s  after_init            246320 kB
 +4.85s  after_model_load      709528 kB
+15.73s  after_model_convert  1815120 kB
+19.65s  after_model_encrypt   2495484 kB
+19.65s  after_input_encrypt   2495484 kB
[+22.95s POLLED PEAK          3389996 kB, VmHWM 3563500 kB]
+93.99s  after_inference      2819296 kB
```

The peak is 3.3 s into inference, inside the first graph operation, the embedding (one-hot matmul
against the whole (28996, 768) table, `crypten/nn/module.py:3306`).

## Census time series, party 0 (`census_pid438870.txt`)

| rss | live tensors | `int64 (28996,768)` | `fp32 (768,768)` | event |
|---:|---:|---:|---:|---|
| 283 MB | 114 MB | 0 | 0 | HF checkpoint opened (mmap, not resident) |
| 1047 MB | 826 MB | 0 | 49 | plaintext model, one copy |
| 1114 MB | 828 MB | 0 | 98 | `from_pytorch` -> `copy.deepcopy(pytorch_model)`: second plaintext copy |
| 2617 MB | 1959 MB | 1 | 98 | `encrypt()` done, int64 shares beside both plaintext copies |
| 2716 MB | 2158 MB | 2 | 98 | embedding matmul begins |
| 3000 MB | 2300 MB | 3 | 98 | |
| 3264 MB | 2470 MB | 4 | 98 | at the peak |

- 2617 MB -> peak: RSS +647 MB, live tensors +511 MB, of which 510 MB are three more
  `int64 (28996,768)` copies from the Beaver matmul (weight share, mask `b`, `y - b`, revealed
  `delta`).
- At the peak: plaintext float32 weights 866500608 B = 2 x 433250304, plus 143.1 MB of plaintext
  float32 tracing activations. No plaintext weight is read after encryption.

## Ranking at the peak

| object | bytes | frac of the 3594240 kB peak |
|---|---:|---:|
| plaintext float32 model weights, 2 copies | 866500608 | 0.235 |
| `int64 (28996, 768)` word-embedding shares, 4 copies | 712605696 | 0.194 |
| plaintext float32 tracing activations | 143130624 | 0.039 |
| `int64 (768, 768)` attention weight shares, 49 | 231211008 | 0.063 |
| `int64 (1, 128, 28996)` one-hot temporaries, 6 | 178151424 | 0.048 |

Grouping per README.md (what one function allocates and frees as one thing); copy counts are kept.
