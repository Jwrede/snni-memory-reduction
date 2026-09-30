# m19r_park_boot_layer: measured one position early

Job 46884689, 2026-09-24/25, `zen4`, 16-core cpuset, shared node, two layers, T=16, rc=0, wall
144,467 s, gate dc043a0b, 4 `LEVER|park_boot_layer`. `levers.env`, `logs/` beside this file.

```
m18_copy_after_switch   72,815,488 kB   69.44 GiB   (high-water in the Layer-1 attention; Layer-0 bootstrap3 69.4)
m19r_park_boot_layer    72,693,324 kB   69.33 GiB   -0.17%   (peak: Layer-1 attention)
```

- Layer-0 bootstrap3 window 69.4 -> 53.8 GiB; global peak in the Layer-1 attention (untouched).
- m18's table (attention, inside a head's softmax bootstrap): softmax phase keys (live), attention
  rotation keys (idle during the softmax bootstrap; paid fix: park), X (live). Position 19 is
  `m19_att_keys_park_softmax`; park_boot_layer is position 20 (build-md5 proof: m20s on the m19s image
  against the m20r binary).
- Superseded order: same pair published as m19 park_boot_layer (plateau), m20 att_keys_park_softmax.
