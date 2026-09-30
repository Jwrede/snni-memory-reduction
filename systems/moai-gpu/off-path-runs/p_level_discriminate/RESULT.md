# p_level_discriminate: discrimination test, one mod-switch fewer

Job 45755949, `gpuh200`, 2026-08-04. Diagnostic (deliberately broken image): campaign tree with
`test_single_layer.cuh:868`'s eleven post-bootstrap-2 mod-switches reduced to ten; same seed and
workload. Condition of the campaign decision of 2026-08-03 for the structural gate.

Stability was known (md5 `7d2520932ba9`, three runs, both seeds). Reference (`det_check/run1`) against
the probe, common prefix:

```
  Modulus chain index before attention block: 14        identical
  Modulus chain index for the result: 0                 identical
  Modulus chain index after bootstrapping: 20           identical
  Modulus chain index after layernorm: 0                identical
  Modulus chain index after bootstrapping: 20           identical
- Modulus chain index before intermediate linear: 9    +  ... : 10
- Modulus chain index after inter layer: 8             +  ... : 9
- Modulus chain index for gelu: 1                      +  ... : 2
- Modulus chain index after final layer: 0             +  ... : 1
```

- PASS: divergence from the first reachable line, one level, persisting.
- The probe then ran out of VRAM (`PROGRAM_EXIT_CODE=1`, `cudaMallocAsync ... out of memory` at
  `cuda_wrapper.cuh:210`, after 2829 s): one level up = 768 MiB more per 768-ciphertext array.
- 9 of 12 trace lines compared. Completing counterpart: `../p_level_discriminate2/`.
- Instrument defect: `REF=$(ls A B 2>/dev/null | head -1)` under `set -e` and `pipefail` exited 2 before
  the comparison (SLURM `FAILED exit 2`, no verdict); verdict recovered by hand. Fixed in
  `impl/p_level_discriminate.sbatch` (shell globbing, `set +e` around the search, short-trace handling).
