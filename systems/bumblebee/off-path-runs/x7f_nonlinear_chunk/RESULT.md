# x7f_nonlinear_chunk: row-blocked non-linearities at b8

| | |
|---|---|
| jobs | 47061625-27 (runs), 47061628-30 (derives), 2026-09-30 |
| machine | PALMA 36-core Skylake, 95 GB; `BB_MAX_CONCURRENCY=1` (`b8`), `OMP_NUM_THREADS=16` |
| image | `bumblebee-f_b9_nonlinear_chunk.sif`, from `impl/tree/`: `b4_slice_budget_64m` with faithful operators; softmax and GELU in blocks of 32 of 128 rows (`in_row_blocks`); row reductions unchanged |
| note | the comment above `in_row_blocks` quotes the placeholder line's numbers |

| | b8_ot_instances_1 | x7f | change |
|---|---:|---:|---:|
| peak_host | 1,460,632 kB | 1,043,408 kB | -28.6% |
| replicate spread | 1.77% | 1.61% | |
| wall, median corrected | 1335.0 s | 1528.3 s | +14.5% |
| rank 0 received (transcript) | 2,933,188,586 B | 4,008,306,621 B | +36.7% |
| rank 0 sent | 3,736,865,896 B | 4,294,547,013 B | +14.9% |

- Each non-linear call reopens its OT conversation (fixed cost per opening): more traffic, so not a
  phase B step.
- Locates the 402.7 MB `yacl::Buffer` at `b8`'s peak as the protocol's batch size.
- Placeholder line: -28.9%, +41% received.
