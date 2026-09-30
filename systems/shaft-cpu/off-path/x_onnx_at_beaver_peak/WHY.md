# x_onnx_at_beaver_peak

| | |
|---|---|
| measured | 2026-08-04, 3 replicates, jobs 45757751-53 |
| role | `s4` candidate of the re-derived order |
| change | remove the 867 MB `io.BytesIO` doubling pair (ONNX single export) |
| peak | -52,412 kB, -2.16% |
| spread | 9.76% (min 2,279,084, max 2,516,624 kB) |
| wall | -3.7%, free |
| gate | pass; conformant on pool and object |
| verdict | refused by `make_steps.py`: peak did not move |

| hump | after this change |
|---|---|
| conversion | ~2.28 GB |
| Beaver | ~2.43-2.52 GB |

The fix removes the buffer from the conversion hump; the binding hump is the Beaver hump. Combined
with the spill in `s4_onnx_and_spill` (see `x_spill_at_convert_peak`).
