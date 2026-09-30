# p_input2: second input

Diagnostic, not a step. Question: does a party's resident peak depend on the input values.

| | |
|---|---|
| machine | `normal`, 36-core Skylake, 95 GB, exclusive |
| replicates | 3 per configuration, same session |
| second input | `impl/levers/probe_input_seed_cpu.py`: torch seed of the input token IDs shifted by `SHAFT_INPUT_SEED_OFFSET=0x1234`; CrypTen seeds unchanged; prints `INPUTSEED|`; refuses offset 0 |
| check | first output logit at `s0`: 0.257 against 0.555 |
| step environments | `stepenv/` |

| run | configuration | input |
|---|---|---|
| `p_rerun_s0` | `s0_default` | published |
| `p_input2_s0` | `s0_default` | second |
| `p_rerun_s8` | `s8_encrypt_chunk` | published |
| `p_input2_s8` | `s8_encrypt_chunk` | second |

Published peak (median replicate, maximum over parties, recorder table subtracted), kB:

| configuration | same-session rerun | second input | change | rerun spread |
|---|---:|---:|---:|---:|
| `s0` | 3,594,368 | 3,578,716 | -0.44% | 0.49% |
| `s8` | 910,380 | 926,540 | +1.8% | 15.9% |

Both changes are inside the rerun spread. At `s8` the peak is a transient inside an encoder layer
(published replicate spread 11.9%).
