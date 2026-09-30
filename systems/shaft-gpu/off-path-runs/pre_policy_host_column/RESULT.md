# pre_policy_host_column: withdrawn host column

Withdrawn numbers of an earlier ordering, not a refuted step.

```
                    old host        old spread     new host        new spread
s0_default          3,306,324 kB      16.79%       2,952,288 kB      14.83%
s1_limb_loop        2,929,464         14.28%       2,935,336          2.25%
s2_embed_chunk      2,939,576          6.60%       2,940,520          0.19%

old:  s1 gains -376,860 kB on host, i.e. -11.4%
new:  the whole line is FLAT, -0.40% end to end
```

- VRAM unchanged and exact in all runs: 6,971,392 / 2,406,400 / 1,546,240 kB (4.509x).
- Cause: the old `s0` value was a high-mode replicate. New `s0` replicates 2,923,028 / 2,956,388 /
  3,361,452 kB.
- A claimed gain below the baseline's spread is not a measurement.

## Harness defect found on the way

| | |
|---|---|
| defect | levers were command-line arguments to `impl/run_line.sh`; the shared `run_step.sh` ran the baseline under all three step names (`RUN_META` of `s2_embed_chunk`: `SHAFT_LIMB_LOOP=<unset>`) |
| why undetected | `check_lever` verifies requested levers; an unset variable requests nothing |
| fix | `impl/stepenv/{s0_default,s1_limb_loop,s2_embed_chunk}.env` (values from the published RUN_METAs); `shaft_gpu_campaign.sbatch` refuses to start without a declaration; `tools/palma/deploy.sh` ships `stepenv/*.env` |
| void runs | `results/.void-nolevers-*` on the cluster |

`s0` spread 14.83% (one replicate at 3.36 GB, two at 2.94 GB); three replicates cannot tell a mode
from a tail.
