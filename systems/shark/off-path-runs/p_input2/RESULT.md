# p_input2: second input

Diagnostic, not a step. Question: does a party's resident peak depend on the input values.

| | |
|---|---|
| change | `fill_span(x, salt, 65536)` salt `0xB0B0B` instead of `0xA11CE` (`impl/src/p_input2_s0.cpp`, `impl/src/p_input2_s5.cpp`); dealer randomness unchanged |
| machine | `zen2-128C-496G`, 16 threads, recorder `c1-pregit-20260730` |
| replicates | 3 each |
| check | `fnv1a64` `a6aa29436dd1b69a` against `4a298ab18dac1ac5` at `s0` |

Published peak (median replicate, maximum over online parties, recorder table subtracted), kB:

| configuration | published | second input | change | spread (published / second input) |
|---|---:|---:|---:|---:|
| `s0` | 58,089,352 | 58,090,952 | +0.003% | 0.011% / 0.009% |
| `s5` | 469,792 | 469,912 | +0.026% | 0.282% / 0.084% |

Both changes are inside the spread. Freeze-corrected median online time, second input against
published (measured on another day): 81.8 against 80.3 s at `s0`, 96.1 against 95.0 s at `s5`.
