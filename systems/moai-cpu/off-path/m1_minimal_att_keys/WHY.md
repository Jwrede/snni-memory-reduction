# m1_minimal_att_keys (old chain): withdrawn 2026-09-03

| | |
|---|---|
| was | first step of m0 -> m1_minimal_att_keys -> m2_pool_threshold |
| measured | -5.2% on the stock pool |
| reason | reordered pool-first (m0 -> m1_pool_threshold -> m2_minimal_att_keys). On the stock pool the key bank is dead at the peak (inside `sealpool:retained`) and cannot be selected; after the threshold pool it is a live rank-1 object |
| now | `m2_minimal_att_keys` (-7.4%, gate bit-identical to m0) |

Logs kept for the -5.2% stock-pool measurement.
