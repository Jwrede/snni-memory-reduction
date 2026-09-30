# m6_att_output_release: withdrawn

Job 46815758 (`p_m6_refreeze`), 2026-09-20/21, `zen4`, 16-core cpuset, shared node, two layers, T=16,
rc=0, wall 85,517 s. Published in the first regeneration of 2026-09-21, withdrawn the same day.
`levers.env` beside this file (plus a header line and the corrected object name).

```
m5_boot_layer_release   209,357,328 kB   199.64 GiB
m6_att_output_release   208,919,296 kB   199.23 GiB      -0.21%
```

- `LEVER|att_output_release|released=768` twice; gate bit-identical to m0 (md5 712aece2); inside the
  0.53 pp spread.
- m5's table shows 15.5 + 15.5 GB at `test_full_scheme.hpp:1223/1226` in `._omp_fn.11`, read as the
  attention output and its concatenation. Lines 1225/1226 are `enc_ecd_x[k] = rtn2[k]` and
  `enc_ecd_x_copy[k] = rtn2[k]` in the bootstrap4 loop (line 1220): the next layer's input, live.
  `att_output` (line 703) is 1.6 GB at that peak.
- At m5's peak no object has a free fix (key bank: phase split, paid; pool aggregate: the knob;
  bootstrap temporaries and output pair: live; attention keys: park, paid).
- The phase-split image was built on this image: `steps/m6_phase_split/levers.env` (that line) lists
  `patch_att_output_release.py` as a carried change without attributed reduction.
