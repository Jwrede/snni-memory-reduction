# BumbleBee: paid phase, b4 to b8

README.md rule 6: after the free fixes, paid fixes ranked by memory per run time, until the target.
Values from `steps.csv`.

## Free phase exhausted

| claim | evidence |
|---|---|
| free object-level fixes taken | `b1`, `b2`, `b3`: -44.1%, -12.8%, -7.8%, free, gated; `b4` (encode slice budget) -3.0%, marginal |
| SEAL pool policy does not help | `off-path-runs/x4f_seal_pool_new`: `MMProfNew` on `b3` raises the peak |
| OT instances are in use | `b5` costs +15.0% run time at a 0.7% spread |

## The line

| step | peak kB | W | wall s | runtime vs predecessor | cost | OT buffers at the peak |
|---|---:|---:|---:|---:|---|---:|
| b3_weight_defer | 6,962,708 | 17.11 | 374 | +0.6% | free | 5368.8 MB |
| b4_slice_budget_64m | 6,751,264 | 16.59 | 388 | +3.2% | marginal | 5368.8 MB |
| b5_ot_instances_8 | 3,889,704 | 9.54 | 423 | +15.0% | paid | 2684.4 MB |
| b6_ot_instances_4 | 2,505,492 | 6.13 | 539 | +29.2% | paid | 1342.2 MB |
| b7_ot_instances_2 | 1,796,776 | 4.38 | 814 | +51.0% | paid | 671.2 MB |
| b8_ot_instances_1 | 1,460,632 | 3.55 | 1342 | +66.8% | paid | 335.6 MB |

`b0` -> `b8`: 15,480,820 -> 1,460,632 kB, 10.60x, `W` 38.09 -> 3.55. `b8` is the target's OT term
(one instance).

## Transcript

`OMP_NUM_THREADS` stays 16. Rank 0, bytes sent, median of three:

| step | transcript vs predecessor | own replicate spread | instances removed | per instance |
|---|---:|---:|---:|---:|
| b1 to b4 | -8.6e-7 to +1.0e-6 | 1.1e-6 to 3.5e-6 | -- | -- |
| b5 | -1.00e-3 | 1.3e-7 | 8 | 0.47 MB |
| b6 | -6.44e-4 | 1.9e-6 | 4 | 0.60 MB |
| b7 | -3.23e-4 | 2.3e-6 | 2 | 0.60 MB |
| b8 | -1.26e-4 | 1.6e-6 | 1 | 0.47 MB |

Gate: worst 5.58e-3 against 7.3e-3 (baseline replicates differ by up to 4.62e-3).

## OT rows against source

| OT instances | measured, both `BasicOTProtocols` rows | predicted (`instances x 2 x 160 MiB`) | `yacl::Buffer` row |
|---:|---:|---:|---:|
| 16 | 5120 MiB | 5120 MiB | 402.8 MB |
| 8 | 2560 | 2560 | 402.7 |
| 4 | 1280 | 1280 | 402.7 |
| 2 | 640 | 640 | 402.7 |
| 1 | 320 | 320 | 402.7 |

The `yacl::Buffer` row (`std_function.h:291 _M_invoke <- buffer.h:44`) is the OT layer's per-call
working set across a non-linear gate; at `b8` it is the largest object (27%).
