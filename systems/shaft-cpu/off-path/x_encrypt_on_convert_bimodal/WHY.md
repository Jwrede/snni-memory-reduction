# x_encrypt_on_convert_bimodal

| | |
|---|---|
| measured | 2026-08-04, 3 replicates, jobs 45764640-42 |
| role | `s5` candidate of the re-derived order |
| change | encrypt each ONNX initializer in the loop that creates it, plus early release of the caller's plaintext model |
| peak | -106,652 kB, -4.89% |
| spread | 9.867% (min 2,097,424, max 2,313,132 kB) |
| gate | pass |
| verdict | refused by `make_steps.py`: peak did not move |

- Two humps of similar height: the conversion hump falls, the Beaver hump does not; each replicate
  peaks in one of them.
- Declared target `_multiarray_umath...+0x13ab14` (865.8 MB) is unchanged in all three replicates
  (`the peak fell 106652 kB but the attacked object's row fell only -4 kB`).
- With the spill active since `s4`, this site holds the read-back parameter set (866,494,480 B). The
  conversion-time plaintext copies are a second object of the same site at another instant (merge
  hazard).
