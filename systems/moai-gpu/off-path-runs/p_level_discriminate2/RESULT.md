# p_level_discriminate2: discrimination test, one mod-switch more

Job 45763737, `gpuh200`, 2026-08-04. Same tree and seed; `test_single_layer.cuh:868` raised from
eleven to twelve mod-switches (lower level, no OOM possible).

```
NOTE: the probe emitted 8 of the reference's 12 trace lines (rc=134); comparing the
      common prefix. A divergence inside it is still a divergence.
VERDICT: PASS -- the trace moved under a structural change. First divergence:
6,8c6,8
< Modulus chain index before intermediate linear: 9      (reference)
< Modulus chain index after inter layer: 8
< Modulus chain index for gelu: 1
---
> Modulus chain index before intermediate linear: 8      (probe)
> Modulus chain index after inter layer: 7
> Modulus chain index for gelu: 0
```

- Reference md5 `7d2520932ba9...`, probe md5 `4a511b2da0d8...`.
- Aborted (`PROGRAM_EXIT_CODE=134`, signal 6) after 8 of 12 lines: modulus budget exhausted. In both
  probes the trace diverges before the program fails.
- Instrument defect: the success branch ended `diff ... | head -12`; `diff` exits 1 on difference,
  `pipefail` and `set -e` killed the script before `exit 0` (SLURM `FAILED exit 1`). Fixed with
  `|| true`.
