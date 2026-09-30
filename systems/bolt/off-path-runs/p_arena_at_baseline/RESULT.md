# p_arena_at_baseline: MALLOC_ARENA_MAX=1 at the baseline

Jobs 45790307-09 (3 replicates, full instrumentation, through `run_step.sh`), 2026-08-04.

Question: is there a free lever that would have to precede the paid `s1_stream_weights` (rule 6)?
Earlier campaign: `MALLOC_ARENA_MAX=1` plus trim, 112.6 -> ~95-101 GiB.

| | median peak (kB) | replicate spread | median wall (s) |
|---|---:|---:|---:|
| `s0_default` | 112,806,764 | 8.909% | 398 |
| this probe (`MALLOC_ARENA_MAX=1`) | 104,666,604 | 2.8% | 722 |

- Peak -7.2%, inside the baseline's 8.9% spread: not established.
- Wall +81%: one arena serialises allocation at 16 threads.
- Not free, so the line's order stands; not a paid step either (peak not moved beyond the spread).
- Baseline peaks 105,407,064 / 112,809,840 / 115,443,896 kB: the widest spread of any system.
