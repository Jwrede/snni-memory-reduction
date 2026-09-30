# BumbleBee: off-path

## Status: the line is closed

| | |
|---|---|
| rows | `b0`-`b8`, 3 replicates each, faithful softmax and LayerNorm (`MANIFEST.md`, Workload) |
| peak | 15,480,820 -> 1,460,632 kB, 10.60x |
| `W` | 38.09 -> 3.55 |
| cost order | `b1`-`b3` free, `b4` marginal, `b5`-`b8` paid (`PAID-PHASE.md`) |
| gate, conformance | every step gated and conformant on pool and object |
| placeholder-operator line | not in the repository; peaks within the replicate spread at every step |

`b8` runs one OT instance, the target's OT term. At `b8`'s peak:

| object | size | disposition |
|---|---:|---|
| `yacl::Buffer` under `_M_invoke` (OT per-call working set) | 402.7 MB, 27% | constant over the instance sweep; its lever (`x7f_nonlinear_chunk`) fails the transcript condition |
| Ferret OT buffers, one instance | 2 x 167.8 MB | target's OT term |
| SEAL pool, across the objects it serves | ~268 MB (median replicate), against an 80.2 MB HE term in the target | policy lever `x4f_seal_pool_new` +14.4%; largest tenant (encode slice) capped by `b4` |
| located-but-unnamed | 17.2% | not a target |

Open: a chunking of the non-linear calls that does not reopen the OT conversation.

## Trades

### Ferret LPN buffer shrink (security)

`off-path-runs/x_lpn4m_faithful`, at `b8`:

| | |
|---|---|
| parameters | `(n, k, t) = (3,997,696, 226,720, 976)` against stock `(10,485,760, 452,000, 1,280)` |
| peak | 1,460,632 -> 1,257,028 kB, -13.9% (source arithmetic of two adapters within 0.4%) |
| run time | +0.6% |
| status | changes the LPN instance the OT security rests on; no attack-cost estimate at these parameters; not recommended |

## Diagnostics

| run | measured | why not a step |
|---|---|---|
| `x4f_seal_pool_new` | SEAL `MMProfNew` on `b3`: peak +14.4%, run time +5.6% | worse |
| `x7f_nonlinear_chunk` | non-linearities in 32-row blocks at `b8`: peak -28.6%, received +36.7% | changes what the protocol exchanges |
| `x_lpn4m_faithful` | above | security trade |
| `p_seq197_faithful` | `b0`, `b7`, `b8` at 197 tokens, no recorder: 10.60x -> 3.24x; `b8` +3.0% over `b7` | other operating point |
| `x_net_grid_proxy` | `b4` against `b8` on a 3 x 4 bandwidth x RTT grid: run-time ratio 4.0x (1 Gbit/s, 0 ms) to 1.3x (50 Mbit/s, 80 ms); peak 6.89-7.01 / 1.49-1.59 GB; +5.2% at one instance from 1 Gbit/s to 50 Mbit/s | one run per cell, no recorder |
| `p_input2` | `b0`, `b8` with input seed 7777: +0.06%, +0.15% | inside the spread |

## Pinned setup

| | |
|---|---|
| threads | 16 per party (`MANIFEST.md`); the OT term is `instances x 320 MiB` (`yacl_ote_adapter.h:58,114`) and `BB_MAX_CONCURRENCY` sets the instances (16 at the baseline) |
| earlier campaign lead | 11.5 of 16 cores used at T=8, 11.1 at T=16: communication-bound (older instrument) |
| gate observable | added to the benchmark (output was revealed and dropped) |
| libstdc++ | linked dynamically so `LD_PRELOAD` can interpose `operator new` |

Both setup changes apply to every step and do not change what the protocol allocates.
