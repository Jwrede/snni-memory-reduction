# p_concurrency: concurrency donor family

Jobs 45800839-41 (`OMP_NUM_THREADS=1`) and 45800918-20 (`max_concurrency=1`), 3 replicates each,
full instrumentation, through `run_step.sh`, 2026-08-04/05.

| | |
|---|---|
| donor | BumbleBee `b5`-`b8` (`ot_instances_8/4/2/1`): lower parallelism where the binding buffer is per instance |
| PUMA knobs | `OMP_NUM_THREADS` (pinned 8), SPU `max_concurrency` (unset in `3pc.json`: default) |
| not transferable | BumbleBee `b4_slice_budget_64m` corrects a budget its own `b1` introduced |

| | median peak (kB) | vs baseline |
|---|---:|---:|
| `s0_default` | 7,278,356 | -- |
| `OMP_NUM_THREADS=1` | 7,299,968 | +0.30% |
| `max_concurrency=1` | 7,282,492 | +0.06% |

- Baseline spread 0.331%: both no-ops. Earlier campaign: `02_thread`, `08_maxconc` no-ops.
- PUMA's peak is one serialization arena in the receiving party (77%), not a per-thread family.
