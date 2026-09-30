# ARION: off-path

## Status: the line runs to `a8`

One run per step, Go heap census inside the measured run (`SNNI_COMBINED=1`), free fixes first
(rule 6). Objects keyed by the first `Arion/` frame (`tools/attrib_go/join_residency.py
--program-root Arion/`); leaf-keyed tables beside them as `objects_*.leaf.jsonl`. Target 8,704,000 kB.

| step | peak (kB) | `W` | cost | node |
|---|---:|---:|---|---|
| `a0_default` | 564,255,692 | 64.83 | baseline | exclusive |
| `a1_btpeval_shallowcopy` | 316,007,184 | 36.30 | free, wall -28.0% | exclusive |
| `a2_activation_input_release` | 276,492,432 | 31.76 | free, wall -0.9% | exclusive |
| `a3_gogc25_direct` | 176,683,776 | 20.30 | free, wall +2.4% (declared paid) | exclusive |
| `a4_threads16` | 142,815,504 | 16.41 | paid, wall +130.7% | exclusive (job 47026510) |
| `a5_qkv_head_release` | 131,619,236 | 15.12 | free by mechanism | shared |
| `a6_v_rotation_defer` | 128,536,292 | 14.77 | free by mechanism | shared |
| `a7_minimal_galois_keys` | 118,691,008 | 13.64 | free by mechanism | shared |
| `a8_threads8` | 113,268,536 | 13.01 | paid, wall +53.3% | shared |

- 4.98x. Wall: program clock minus freeze and census pauses. Shared nodes (700 GB) carry co-tenant
  noise; their cost is declared by mechanism in `levers.env`. Exclusive replicates of `a5`, `a6`
  (47044193, 47044194) were cancelled before starting. No trades.
- Census instants: a dump is written at a new maximum at least 5% above the previous dump. Tables of
  `a1` and `a3`-`a8` lie 1.8 to 4.1% below their HWM (flagged by `make_steps.py`):

| step | last dump | peak |
|---|---|---|
| `a5` | start of layer 1's attention, before any head's columns were released | 837 s and 1.8% later |
| `a6`, `a7` | layer 0 feed-forward window, 4.0% and 2.4% below | start of layer 1's attention |
| `a8` | layer 0 feed-forward window, 3.9% below | same window of layer 1 |

Reruns of `a6`-`a8` with a 0.5% dump step (`SNNI_CENSUS_PERMILLE=5`) are not part of the published line.

## End of the line

- Closed-arm line not carried to completion. The `a8` peak is a plateau of both layers' feed-forward
  windows and the start of layer 1's attention, within 0.1%; no free fix remains.
- 4 workers: would act on all three windows; not taken (a4 +130.7%, a8 +53.3% wall; a8 runs about
  16 hours, 57,441 s of program time at 8 workers).
- Feed-forward tiling (donor MOAI-CPU `m11_tiled_ffn`): would lower only the two feed-forward windows.
- Neither is measured (`steps/a8_threads8/levers.env`).

## Skipped objects

| step | skipped | size | disposition |
|---|---|---:|---|
| `a2_activation_input_release` | `activation.ApproximatePolynomialChebyshevMT.func1` (rank 1 at `a1`) | 218.8 GB | live part (per-worker power basis, results) shrinks only with fewer workers (paid); dead share only with lower GOGC (`off-path-runs/a2_gogc50`: +13% wall) |
| `a6_v_rotation_defer` | `matrix.CiphertextMatricesMultiplyWeightAndAddBiasMT.func1` at `a5` | 33.8 GB | Q/K/V columns `a5` releases head by head; the census fell before any release |
| `a7_minimal_galois_keys` | same row at `a6` | 32.4 GB | at `a6`'s census instant the FFN intermediate; its free fix is on the line (`a2`), tiling is paid |
| `a7_minimal_galois_keys` | `go:runtime_retained` | 28.4 GB | only lower `GOGC` reaches it (on the line at `a3`); costs run time |

## Runs kept as evidence

| run | content |
|---|---|
| `off-path-runs/a2_gogc50`, `a3_gogc25` | GOGC ladder replaced by `a3_gogc25_direct` |
| `off-path-runs/a3_minimal_galois_keys` | minimal keys on `a2`: -7.3% at +22.4% wall (one run each); later `a7` |
| `off-path-runs/a6x_minimal_galois_keys_before_vdefer` | minimal keys on `a5`, with the key-row breakdown |
| `off-path-runs/a4_threads16_shared` | shared-node replicate of `a4` (peak 0.05% apart) |
| `off-path-runs/a3_threads16` | thread step of the superseded ordering |
| `off-path-runs/p_goheap`, `p_goowner`, `p_floor_bench_att` | census, owner view, target bench |

The earlier line (two channels per step, three replicates at the baseline, GOGC 50 before the
activation release) is not in the repository.

## Withdrawn before any run: `a1_dft_one_at_a_time`

`off-path/a1_dft_one_at_a_time/levers.env`.

| | |
|---|---|
| premise | the census's 210.6 GiB under `dft.NewMatrixFromLiteral` is one pair of DFT matrix sets; hold one at a time |
| derivation (`impl/derive_dft_size/`, no run) | one pair is 3.24122 GiB; 210.6 / 3.24122 = 64.98 |
| actual | `pkg/btp/matbtp.go:60` deep-copies the bootstrapping evaluator per goroutine: 64 workers plus the original |
| replacement | `a1_btpeval_shallowcopy` (free; leaves 3.24 GiB) |

Per worker: 3.24 GiB matrices plus ~0.6 GiB `rlwe.NewEvaluatorBuffers` = 3.84 GiB. The earlier
campaign's 242 -> 139 GiB from 64 to 16 threads has this shape (other scope; not recomputable).

## Declared scope

Two of twelve layers (MANIFEST.md, `WORKLOAD|` marker): a reduced workload, applied to every step;
absolute peaks are not comparable to twelve-layer systems.

## Known in advance (earlier campaign)

| item | status |
|---|---|
| `GOMEMLIMIT` cap | enabler there (OOM on 251 GiB without it); under a cap `VmRSS` tracks the cap. Not set here (1.5 TB node) |
| thread count | 242 -> 139 GiB from 64 to 16 threads there; a paid step here (`a4`, `a8`); baseline pins upstream's 64 |
| key streaming (`fork/galois_stream.go`) | -5.5% on the floor there; patches lattigo's evaluation-key expansion (third-party cryptography): off-path |
