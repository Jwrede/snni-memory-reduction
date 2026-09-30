# x_lpn4m_faithful: Ferret LPN trade at b8

Off-path (security trade, `../../off-path.md`).

| | |
|---|---|
| jobs | 47061613-15 (runs), 47061616-18 (derives), 2026-09-30 |
| machine | PALMA 36-core Skylake, 95 GB; `BB_MAX_CONCURRENCY=1`, `OMP_NUM_THREADS=16` |
| image | `bumblebee-f_x_lpn4m.sif`, from `impl/tree/`: `b4_slice_budget_64m` with faithful operators; `libspu/mpc/cheetah/ot/yacl/yacl_ote_adapter.h` `lpn_param_{3997696, 226720, 976, ...}` instead of `n = 10,485,760` |

| | b8_ot_instances_1 | x_lpn4m_faithful | change |
|---|---:|---:|---:|
| peak_host | 1,460,632 kB | 1,257,028 kB | -203,604 kB (-13.9%) |
| replicate spread | 1.77% | 0.32% | |
| wall, median corrected | 1335.0 s | 1343.6 s | +0.6% |

Source: adapter buffer `n * 16 B`, 167,772,160 B stock, 63,963,136 B reduced; two adapters shed
207,618,048 B = 202,752 kB. Measured 203,604 kB (within 0.4%).
