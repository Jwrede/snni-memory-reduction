# BOLT p_window, k=1 control

Measured 2026-08-13 on `bigsmp`, 3 runs (jobs 46089721, 46089885, 46090057), image `bolt-p_window.sif`
(binary md5 `a4c2617472303bbbb2aba615d6f750fc`; `s1_stream_weights`: `18a9f64fc4bb1034bd17910b98919f38`).
Each run announced `LEVER|layer_window|k=1`.

`patch_layer_window.py` makes `s1`'s one-layer residency a parameter `k`. At `k = 1` it must
reproduce `s1_stream_weights`.

|  | `s1_stream_weights` (published) | `p_window` at k=1 |
|---|---|---|
| wall | 661 / 673 / 681, median **673** | 679 / 685 / 696, median **685** |
| wall spread | 3.03% | 2.50% |
| peak (kB) | 17,481,140 / 17,506,216 / 17,595,044, median **17,506,216** | 17,496,324 / 17,592,336 / 17,598,972, median **17,592,336** |
| peak spread | 0.65% | 0.59% |
| gate logit0 | [-4.890137, -4.886230] | [-4.894287, -4.888672] |
| gate logit1 | [5.285645, 5.291748] | [5.276123, 5.285400] |

- Peak: +0.49%, inside both spreads: control passes.
- Gate: largest deviation over 18 pairs 1.562e-2 (86.3% of 1.81e-2).
- Wall: +1.8% median gap, inside both spreads (after two runs the ranges did not overlap; the third
  run removed the apparent cost).
- `logit1` offset: every k=1 run below every `s1` run, gap 2.44e-4. The parameterised version enters an
  OpenMP parallel region over one element, which changes thread scheduling and the OT draw order.
  The curve over `k` is referenced to k=1 on this binary.
- Job 46089887 died after 17 s (`FAILED: a bind conflict during the run; this measurement is void`):
  this build binds further ports for its non-linear threads outside the reserved block. Resubmitted as
  46090057.
