# PUMA: off-path

## Status: the line is closed

| step | peak (kB) | `W` | cost | gate |
|---|---:|---:|---|---|
| `s0_default` | 7,318,424 | 19.55 | baseline | |
| `s1_weight_stream` | 3,167,480 | 8.23 | free (-11.7%) | pass |

-56.7%, 2.31x. No trades. Closed arm: a step must be a lever measured on a primary system.

## Donor levers measured

| donor lever | source | result |
|---|---|---|
| per-layer weight streaming (staging the computation) | SHAFT-CPU `x1_per_layer` | transfers: the line's `s1` |
| per-layer weight infeed (staging the transfer) | SHARK `s1` / SHAFT per-layer loading | regression, +1.77% against a 0.344% spread (`off-path/x_layer_infeed/`) |
| release an intermediate after its last consumer | BumbleBee `b2_result_ct_pack_free` -> LLAMA `o2_activation_free` | -12.4% at the baseline, +1.9% after `s1` (`off-path-runs/s2_response_stream_on_ws/`) |
| allocator arena cap (`MALLOC_ARENA_MAX=2`) | LLAMA-Dealer | -2.9%, withdrawn: attacks `anon:` (`off-path/s1_arena_cap/`) |
| concurrency reduction (paid family) | BumbleBee `b5`-`b8` | no-op on both knobs, +0.30% and +0.06% (`off-path-runs/p_concurrency/`) |
| plaintext model release | SHAFT-CPU `s1_plaintext_free` | does not transfer in this form: the model is live at the peak (`off-path-runs/s2_gc_threshold/`) |

- Arm rule: on 2026-08-17 SHAFT-CPU measured per-layer streaming as a primary system
  (`x0_per_layer_base`-`x4_stream_load`, transcript-identical, 11,228,688,384 bytes in 1495 rounds),
  so PUMA's `s1` is a transfer. Earlier campaign: `05_perlayer` "no-op (XLA liveness)", `06_stream`
  "-57%".

## End of the line

| at `s1`'s peak | size | disposition |
|---|---:|---|
| `heap:spu::NdArrayRef::NdArrayRef` | 1125.2 MB, 34.7% | census (93% of the peak): three concurrent 375,054,672 B `response` objects in `distributed_impl.py:RunReturn`, node4 = P2 (the party holding the model); release-after-use donor changes grouping, not amount (five 128-MiB `list` objects and three 106.6 MB `write()` buffers, same total within 876 bytes) |
| `heap:pybind11::bytes` | 1087.3 MB, 33.5% | `s1`'s object (5250.2 MB at `s0`); sized by the largest single infeed (`off-path-runs/p_census/`): the floor of `s1`'s mechanism |

Remaining mechanism (pieces existing one at a time) has no primary-system donor. The binding
process is P2, which excludes driver-side donors.

## Known off-path in advance (earlier campaign)

| candidate | trade | earlier result |
|---|---|---|
| FM64 -> FM32 | precision | broken: int64 embedding indices do not fit FM32; run exited non-zero without output |
| fewer exp/div iterations | accuracy | no gain (transient is size-bound) |
| `experimental_enable_colocated_optimization` | portability (valid only with all parties on one host) | not run |

## Driver

The driver holds a plaintext copy of the model. If it binds the peak, that is reported with the
binding SPU node beside it; it is not dropped from the poll set.
