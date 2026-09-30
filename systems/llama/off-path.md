# LLAMA online: off-path

## Status

| step | peak (kB) | `W` | cost | gate |
|---|---:|---:|---|---|
| `o0_default` | 24,708,836 | 946.06 | baseline | |
| `o1_activation_free` | 24,408,584 | 934.56 | free | pass |
| `o2_operator_scratch_free` | 24,174,052 | 925.57 | free | pass |
| `o3_key_stream` | 958,700 | 36.50 | marginal (+13.8% at 8.4% spread) | pass |

- -96.1%, 25.774x; n=3; rule-ordered 2026-09-29 (`freeze_mode=maxparty`).
- The key stream measured +16.8% at a 5.1% spread when taken first, so the free steps precede it.
  First ordering archived under `/mnt/HC_Volume_105187419/offload/reorder_archive/llama/`.
- Closed arm: only borrowed levers; no local search. `W` 36.5 is far from the floor (below).
- No trades. The setup changes in `impl/patches/` (runnable, gateable online path) apply to every
  step and change no memory behaviour. The dealer phase is excluded by construction (`llama-dealer`).

## Skip dispositions

| skipped by | object | size | disposition |
|---|---|---:|---|
| `o1`, `o2` | key file, `readFile(...)` | 23,909.8 MB, 94.5% of `o0`'s peak | only fix is streaming (`o3_key_stream`), not free: +16.8% first, +13.8% last |
| `o1`, `o2` (declared) | model weights, `Tensor2D<unsigned long>::Tensor2D` | 682 MB | regeneration from seed is inadmissible (below) |

## Weights: refused lever

- The dealer line's `d5` regenerates byte-identical weights from per-layer seeds.
- The online party masks each layer in place before the forward pass (`initializeInferencePartyA`,
  `llama_base.h:166`): `weight` holds `w + r`. Regenerating `w` would be wrong in every matmul.
- Undetectable here: FSS is data-oblivious (wire gate unchanged), output is 98,304 zeros regardless.
- Working, five source-introspection jobs: `off-path-runs/p_weights/RESULT.md`.

## End of the line

`o3_key_stream`, published peak 958,700 kB (leader; median run's frozen capture,
`steps/o3_key_stream/peak-objects.md`):

| object | size | disposition |
|---|---:|---|
| `heap:tensor.h:523:Tensor2D<u64>::Tensor2D` | 680.1 MB, 69.3% | model weights, inadmissible (above) |
| `anon:[heap]` | 198.2 MB, 20.2% | no allocation site |
| everything else, named | ~73 MB, 7.4% | largest single item 31.5 MB |

- The leader's table at its own peak (958,872 kB): 680.1 MB (69.3%), 198.6 MB (20.2%), 7.5%.
- The addressable surface is the third row (at most 7.4% of the peak).
- The LLAMA dealer stopped at an allocator floor with `W` near 1; here the stop is a correctness bar.
- Across `o3`, `anon:[heap]` grew 189.1 -> 198.6 MB while the peak fell 242.6 MB (predicted from
  source 250.0 MB): the freed blocks stay in the brk region.
