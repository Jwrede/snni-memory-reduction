# a2_gogc50: GOGC 50 on a1

| | |
|---|---|
| run | 2026-09-25, `bigsmp`, a1's image, T64, GOGC 50, in-run census |
| peak | 248,137,676 kB, -21.5% against `a1_btpeval_shallowcopy` (316,007,452 kB) |
| program wall minus frozen | 16,441 s against 14,521 s: +13% (free criterion +4%) |
| effect | only rank 1's dead share: activation 218.9 -> 98.1 GB; other owners unchanged |
| disposition | paid; may not precede the free `a2_activation_input_release`. GOGC entered later at 25 (`steps/a3_gogc25_direct`). Evidence for the rank-1 skip in `a2_activation_input_release` |
