# a3_minimal_galois_keys: minimal Galois keys on a2

| | |
|---|---|
| run | 2026-09-26, job 47016081, T64, GOGC 100, on `a2_activation_input_release` |
| peak | 256,181,204 kB, -7.3% against a2 |
| program wall minus frozen | 17,612 s against 14,390 s: +22.4% (one run each) |
| keys row | 23.9 -> 16.1 GB (48-key bootstrap bank stays; 128 residual keys -> 30) |
| peak moved to | layer 1 final LayerNorm / bootstrap 4 |
| donor | MOAI-CPU `m2_minimal_att_keys` |
| disposition | ranked below the worker-count step at the time (0.3 against an estimated 0.4; `a4_threads16` measured 0.15). On the line as `a7_minimal_galois_keys` (-6.5% wall against a6, free by mechanism) |
