# x_m0_att_keys_release_last_layer: release on the stock pool

Job 46995118, 2026-09-24/25, `zen4`, 16-core cpuset, shared node, two layers, T=16, rc=0, wall
71,872 s (`--mem=430G`; a first attempt at 256G, job 46917952, was OOM-killed at 268 GB MaxRSS, kept as
`run1.oom_256G` on the cluster). `patch_att_keys_release_last_layer.py` on the m0 image (definition
beside this file); the line takes it as m5.

```
m0_default                         402,558,476 kB   383.94 GiB
x_m0_att_keys_release_last_layer   398,760,856 kB   380.31 GiB      -0.94%
```

- `LEVER|att_keys_release_last_layer|layer=1|action=released` once; gate bit-identical to m0 (md5
  712aece2); inside the 2.0 pp spread.
- m0's peak is set in layer 0 (bootstrap3 380.3 GiB, then flat 380.2-380.3 through layer 1). The release
  in layer 1 lowers the live set by 17.2 GB (marker `att_keys_released` 379.9) on a pool that returns
  nothing to the kernel.
- After the pool repair (m1) the same release measures -2.38% (m5).
- `steps/m5_att_keys_release_last_layer/levers.env`: `requires: m1_pool_threshold`,
  `refuted_as: x_m0_att_keys_release_last_layer 402558476 398760856`.
