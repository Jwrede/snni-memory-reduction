# BOLT: off-path

## Status: the line is closed

| step | peak (kB) | `W` | cost |
|---|---:|---:|---|
| `s0_default` | 108,850,800 | 805.22 | baseline |
| `s1_stream_weights` | 17,502,624 | 129.41 | paid (+37.7%) |
| `s2_threads_4` | 17,150,684 | 126.8 | paid (+22.0%) |

6.347x, n=3 each. The line ends where the closed arm's donor pool ends.

| candidate | result | disposition |
|---|---|---|
| `s3_threads_1` (`off-path-runs/s3_threads_1/`) | peak +4.21%, wall +57%; named rows unchanged; anonymous mappings +877.8 MB (138 -> 179) | refuted; thread ladder exhausted at 4 |
| `s2_preproc_arrays_free` | removes rank 1 of `s2`'s table, `Linear::params_preprocessing_ct_pt` (5,176.8 MB, 29.5%), whole: -4.67%, 15% less wall | self-derived, not borrowed: off-path on the closed arm |
| `p_seal_pool_new` (job 46124468) | SEAL `MMProfNew`: 17,150,684 -> 16,531,708 kB (-3.61%, 30x `s2`'s spread), gate held; wall 775 -> 1515 s (+95.5%) | paid and refused by rule 6: on published peaks and freeze-corrected walls ~1,010 kB/s (623,076 kB for +618 s) against ~3,130 kB/s for `s2_threads_4` (351,940 kB for +112.3 s) |

Machine: `bigsmp` (r05n06); comparison with `long` in `MANIFEST.md`. Kept on `bigsmp` on 2026-08-19
(closed arm, 11-minute runs); fallback if the last healthy node (r05n08) goes: 2.2 h of compute on
another node type.

## Earlier campaign on this system (other machine and instrument; evidence about mechanisms)

| finding | relevance |
|---|---|
| `MALLOC_ARENA_MAX=1` plus trim: bulk peak 112.6 -> ~95-101 GiB; `MMProfNew`: 138-140 GiB (worse) | measured here as `p_seal_pool_new`: -3.61%, paid |
| weight streaming (per-layer, then per-linear): ALICE 112.6 -> 7.79 GiB | donor lever (SHARK s1 / SHAFT / PUMA), basis of `s1_stream_weights` |
| QKV split | infeasible: QK^T is fused in this implementation and the split does not build |

## Known off-path in advance

| candidate | trade |
|---|---|
| lower OT bitlength, lower GELU/softmax approximation degree | accuracy (a two-value gate could pass it) |
| SEAL parameter changes (`poly_modulus_degree`, coefficient chain, plaintext modulus) | security level (`sec_level_type::tc128`) |

## `x1_layer1`: depth diagnostic

Baseline image with `ATTENTION_LAYERS = 1`, two runs. Question: truncation error or an unpinned
draw source behind the 1.05e-2 replicate disagreement of `s0`.

| | truncating operations | worst pairwise absolute | peak |
|---|---:|---:|---:|
| `x1_layer1`, 2 runs | 6 | 3.42e-3 | 110,741,936 / 108,546,140 kB |
| `s0_default`, 3 runs | 50 | 1.05e-2 | 108,850,800 kB (median of 3) |

- Disagreement falls 3.1x for 8.3x fewer operations: truncation error. Tolerance 1.81e-2 (`MANIFEST.md`); published steps lie within 1.46e-2 of the
  baseline runs.
- The peak barely changes between 1 and 12 layers (inside the 8.9% spread): it is set by
  preprocessing before the layer loop.
