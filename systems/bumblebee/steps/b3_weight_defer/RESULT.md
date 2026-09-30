# b3_weight_defer: first measurement

Measured 2026-07-30 on the placeholder-operator benchmark (jobs 45520565-67, derives 45520568-70,
`express`, policy `2026-07-29.1`), chained behind the `x1_threads1` diagnostic. The published row is
the faithful re-measurement (`steps.csv`).

## Identification

The declaration predicted that the 769.7 MB `yacl::Buffer` site without discriminator holds the
sealed weight shares, and that the row is void if it does not fall.

| object | b2 | b3 |
|---|---:|---:|
| `buffer.h:44 (discriminator 2)`, Ferret OT | 5595.4 MB | 5771.6 MB |
| `buffer.h:44`, the weight shares | 769.7 MB | 77.9 MB |
| `mempool.cpp:145`, SEAL pool | 528.1 MB | 524.2 MB |

77.9 MB = one layer's shares plus the input embedding. Prior size estimates: 679.5 MB (source),
786 MB (marker pair), 769.7 MB (table).

## Step

| | b2_result_ct_pack_free | b3_weight_defer |
|---|---:|---:|
| peak_host | 7,520,936 kB | 6,938,432 kB |
| W_host | 18.48 | 17.05 |
| named_frac_host | 0.9409 | 0.9428 |
| spread_host_pct | 0.134% | 0.45% |
| wall | 307 s | 304 s |

- -582,504 kB (-7.7%); free (`runtime_delta_pct` -0.9%); `object_conformant: yes`.
- `object_vs_unnamed` 14.96.

## Random draws

`hal::seal` draws the sharing randomness; moving twelve seal calls into the layer loop interleaves
them with the protocol's draws.

| | b3 against its predecessor | noise |
|---|---:|---:|
| gate, worst absolute against the baseline | 5.127e-3 | tolerance 7.3e-3; b0's own spread 5.07e-3 |
| transcript, rank 0 sent | 4.41e-7 relative | 1.02e-6 within b3's replicates |

Instrument sensitivity: `x1_threads1` (same image, same day) moved the transcript by 1.53e-3 inside
the gate tolerance (5.630e-3).

## Correction of b2's skip

`b2` skipped `yacl::Buffer::Buffer(long)` as "the Ferret OT buffer, no free fix". Grouping by
function had merged two objects (Ferret OT and weight shares) behind one container constructor.
