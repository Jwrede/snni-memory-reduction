# LLAMA dealer: off-path

## Status

| step | peak (kB) | `W` | cost |
|---|---:|---:|---|
| `d0_default` | 4,654,148 | 66.12 | baseline |
| `d1_select_free` | 2,129,356 | 30.21 | free |
| `d2_mask_stream` | 1,465,604 | 20.77 | free |
| `d3_activation_free` | 1,121,768 | 15.88 | free |
| `d4_truncate_free` | 861,836 | 12.18 | free |
| `d5_gelu_free` | 694,652 | 9.81 | free |
| `d6_ars_free` | 683,016 | 9.65 | free |
| `d7_weight_init_free` | 619,412 | 8.73 | free |

-86.7%, 7.514x; n=3; every gate byte-identical; every step conformant on pool and object. No trades.

```
server.dat  23,003,004,064 B  md5 86f0777f040c9033007ee3276bfe3165
client.dat  23,909,760,144 B  md5 a639da6964fd33269613ca65408ffa34
```

## Where the line stops

| at `d7`'s peak | size | frac |
|---|---:|---:|
| `anon:[heap]` plus three 64-MiB-aligned arena mappings | 503.1 MB | 81% |
| `SlothWrap_dpf` | 62.9 MB | 10.2% |
| `SumOfSquare` | 19.0 MB | 3.1% |
| `Tensor<unsigned long>::allocate` | 11.1 MB | 1.8% |
| everything else named | 8.9 MB | 1.4% |

- `named_frac_host` 0.165, `object_vs_unnamed` 0.12. Removing every named object would leave ~526 MB.
- `SlothWrap_dpf`: the dealer branch frees each key pack after sending; the 62.9 MB is the live set of
  one `#pragma omp parallel for` that generates all packs before sending. Bounding it repartitions
  the loop and changes the key material (`d7_wrap_chunk`).

## The ~500 MB floor: seven mechanisms

Through `d5` the peak was the accumulating key material at the end of the run; ~505 MB stayed
constant while the peak fell 7x. `mallinfo2` per graph node: 141.8 MB in use, 209.2 MB free but held.

| mechanism | measured | verdict |
|---|---|---|
| `malloc_trim(0)` per graph node | -0.003% peak, +11% wall | free chunks interleaved with live ones; few whole free pages |
| `MALLOC_MMAP_THRESHOLD_=131072` | -0.34% (inside noise) | not large allocations drifting into the arena |
| `MALLOC_ARENA_MAX=1` | +0.004% peak, +32% wall | memory relocates into the main arena |
| recorder floor 64 KiB -> 4 KiB | attribution 15.8% -> 16% | not small allocations under the floor |
| jemalloc 5.2.1 via `LD_PRELOAD` | +174% memory, -46% wall | 1.71 GB against glibc's 624 MB, keys identical |
| exact-size buffer pool in `Tensor::allocate` | +54.7% memory | glibc reuses freed chunks across sizes; a same-size pool holds one buffer per size |
| `OMP_NUM_THREADS` 16 / 8 / 4 / 1 | peak constant within 0.3%, all keys differ | thread count is part of the seed |

`d6` had pushed the end-of-run ramp below a startup transient (model weights); `d7` removed that
transient (683,016 -> 619,412 kB), so the peak is again the end-of-run phase these runs describe.

## Rejected as steps

| run | result | reason |
|---|---|---|
| `d3_alloc_trim` | `malloc_trim` per node: -0.003%, +11% wall | peak unchanged |
| `d7_tensor_pool` | exact-size pool: +54.7% | regression |
| `d7_wrap_chunk` | `SlothWrap_dpf` key generation in 8 blocks: key files 23003004064 B, md5 `67b5cf34...` against `86f0777f...`; saved 2.6% | gate fails: `keyGenWrapDPF` draws from `prngs[omp_get_thread_num()]` via `splitShare` |
| `d7_sumsq_free` | missing `delete[] squares` (24 live buffers at the end-of-run peak) | +0.04% against the then-binding startup peak |

## Withdrawn ordering violations (instrument)

Each was found by building the "corrected" alternative, which then gave the smaller reduction.

### Recorder floor (`d3`)

`d3t_truncate_free` on `d2`: -259,624 kB against the activation lever's -343,836 kB.

| floor | largest | second | attributed |
|---|---|---|---:|
| 1 MiB | TruncateReduce 264.8 MB | Tensor::allocate 129.5 MB | 44% |
| 256 KiB | TruncateReduce 264.8 MB | Tensor::allocate 253.5 MB | 56% |
| 64 KiB | Tensor::allocate 366.4 MB | TruncateReduce 264.8 MB | 66% |
| 16 KiB | Tensor::allocate 366.4 MB | TruncateReduce 264.8 MB | 66% |

`Tensor::allocate`: 2715 allocations, 35% touched, none a megabyte. `drops` 0, peaks within 0.03%.
Campaign floor set to 64 KiB.

### Ranking granularity (`d5`)

`d5a_ars_free` on `d4`: -74,052 kB against the GELU lever's -167,184 kB.

| | per site | per function |
|---|---|---|
| SlothGelu | 34.6 MB x 5 rows | 173.2 MB |
| SlothARS | 76.9 MB | 76.9 MB |
| SlothWrap_dpf | 62.9 MB | 62.9 MB |

`make_steps.py` groups sites by function before ranking.

## Weight tensors: two wrong attempts

| attempt | result | actual cause |
|---|---|---|
| A, `d3_weight_deferred` (release at init, recreate at the matmul) | segfault after 1 s | weights are needed by the up-front masking (`llama_base.h`, `initializeInferencePartyA`) |
| B, streaming the masking | keys differed | own bug: `counter[owner - SERVER]` with `owner = 2`, `SERVER = 2` is `counter[0]`; the patch used `counter[1]` |

With the index fixed the keys are byte-identical: `d2_mask_stream`, 2.03 -> 1.40 GiB. The masks are a
deterministic function of a running counter, restored per layer. The earlier campaign's defective
lazy-weight step measured 1,466,192 kB; the correct lever 1,465,928 kB (0.018% apart).

## Coverage

64 KiB floor, grouped by function. The decomposition names 89% of the peak at `d0` and 39% at `d4`;
the rest is published as `anon:residual`.
