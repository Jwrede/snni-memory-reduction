# p_input2: second input

Diagnostic, not a step. Question: does a party's resident peak depend on the input values.

| | |
|---|---|
| change | secret input from seed 7777 instead of 9999 (`make_secret` in `op_bench.cc`; log `SEEDED|mt19937|weights=100..1200|input=7777`) |
| overlays | `overlay/p_input2_b0`, `overlay/p_input2_b4` (the `b4` tree run with `BB_MAX_CONCURRENCY=1`, as `b8`); image recipes in `def/` |
| machine | `normal`, 36-core Skylake, 95 GB, exclusive, 16 threads |
| replicates | 3 each |
| check | first output value -0.1766 against -0.0962 at `b0` |

Published peak (median replicate, maximum over parties, recorder table subtracted), kB:

| configuration | published | second input | change | spread (published / second input) |
|---|---:|---:|---:|---:|
| `b0` | 15,480,820 | 15,489,880 | +0.06% | 0.19% / 0.17% |
| `b8` | 1,460,632 | 1,462,768 | +0.15% | 1.77% / 2.87% |

Both inside the spread. Freeze-corrected median wall (measured on another day): 335.1 against
341.6 s at `b0`, 1,347.8 against 1,335.0 s at `b8`.
