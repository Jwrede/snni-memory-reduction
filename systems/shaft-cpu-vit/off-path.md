# SHAFT-CPU-ViT: off-path

## Status

| row | host peak (kB) | `W` | spread | cost | gate |
|---|---:|---:|---:|---|---|
| `v0_default` | 3,293,128 | 39.69 | 2.979% | baseline | |
| `v1_endpoint` | 1,171,276 | 13.06 | 7.244% | free | equivalent |

- Reduction 2.81x, against SHAFT-CPU's 4.26x on BERT.
- The line ends by construction: two rows, the BERT stack unchanged. A ladder derived from ViT's own
  tables would answer a different question.
- `W` differs from SHAFT-CPU's because the targets differ in kind (`MANIFEST.md`).

## Earlier campaign, for reference

| | earlier campaign | this line |
|---|---:|---:|
| baseline, 197 tokens | 3,198,080 kB (polled VmRSS) | 3,293,128 kB (VmHWM, maximum over parties); 3,199,688 kB in the runs before 2026-09-29 |
| endpoint `perflayer` | 1,011,028 kB (poller maximum) | not comparable (`VmHWM` is never lower) |
| endpoint `perflayer + malloc_trim + T8` | 632,856 kB | allocator knob, an operating point here (`../shaft-cpu/off-path-runs/p_alloc_s8`) |

## Trades

None. A different ViT size, patch grid, head or token count is a different operating point and
needs its own system.

## Evidence

`../shaft-cpu/off-path-runs/vit_transfer/RESULT.md`; equivalence test and output in
`steps/v1_endpoint/impl/`.
