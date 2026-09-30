# x_spill_at_convert_peak

| | |
|---|---|
| measured | 2026-08-04, 3 replicates, jobs 45757936-38 |
| change | spill all 201 parameter shares |
| peak | 2,482,984 -> 2,483,364 kB, +380 kB |
| spread | 0.386% |
| gate | pass |
| verdict | refused by `make_steps.py`: peak did not move; `object_conformant=no` (largest object at `s3`'s peak was `python3.10+0x13cc27`, the ONNX export buffer) |

| | conversion hump | Beaver hump | published peak |
|---|---:|---:|---:|
| `s3_trace_nograd` | ~2.48 GB | ~2.52 GB | 2,482,984 kB |
| + ONNX single export (`x_onnx_at_beaver_peak`) | ~2.28 GB | ~2.52 GB | 2,430,572 kB, spread 9.76% |
| + spill (this run) | ~2.48 GB | lowered | 2,483,364 kB, spread 0.386% |

Each fix lowers one of two co-binding humps. `s4_onnx_and_spill` carries both.
