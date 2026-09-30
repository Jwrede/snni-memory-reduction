# p_batch_check_only: result

Jobs 45774969-71 (3 replicates, through `run_step.sh`), derives 45774972-74, 2026-08-04, against the
online batch-check image alone.

| | median peak (kB) | n | spread |
|---|---:|---:|---:|
| `d0_default` (no lever) | 868,908 | 3 | 0.014% |
| this run (batch-check mechanism only) | 869,044 | 3 | 0.015% |
| `d1_layer_weights` (both mechanisms) | 259,992 | 3 | 0.112% |

- +136 kB (0.016%) against a 0.014% baseline spread: inert. `d1`'s reduction is the weight
  mechanism alone.
- The borrow is within one system (same binary, other role), linked under the step's name.
