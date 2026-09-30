# m2_x_copy_release: release fired, peak unchanged

Job 46381537, 2026-08-24, `zen4` exclusive, two layers, T=16, rc=0, wall 38,690 s. Withdrawn before
publication. `levers.env` beside this file, unedited.

```
m1, re-measured under the chain recorder   380,386,768 kB
m2_x_copy_release                          379,933,052 kB      -0.12%
```

- `LEVER|x_copy_release|released=768`; gate bit-identical over 7,680 values.
- Prediction -2% to -6%; refutation bound 0.15% then, 0.53 pp later. Refuted.
- Risk named in `levers.env`: "the gain is that the FFN's allocations reuse the freed blocks instead of
  growing the pool, and that depends on the size classes matching. The copy sits at a mod-switched
  level and the FFN allocates at others."
- At `m0`'s peak (`../p_prm0/`): peak 402.3 GB = live 239.1 GB (59%) + retained by the pool 163.2 GB
  (41%). A release converts live memory into retention.
- `m1_minimal_att_keys` works because it prevents growth instead of releasing.
