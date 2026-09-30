# s2_param_stream: placement, not a reduction

Jobs 45635371-73, 3 replicates, gate pass, MIG slice, earlier ordering. Declarations conformant:
`pool_attacked: vram`, `pool_skipped: host`, `object_attacked: mpc.py:165:cuda`.

| | s1_limb_loop | s2_param_stream | |
|---|---:|---:|---|
| VRAM peak, kB | 2,406,400 | 1,695,744 | -710,656, -29.5% |
| host peak, kB | 3,412,564 | 3,500,296 | +87,732, +2.5% |
| `W_vram` | 10.01 | 5.92 | |
| host spread, 3 runs | 15.591% | 0.325% | |
| wall, corrected median | 40.5 s | 42.3 s | +4.6%, inside the spread |

`cross_pool=yes`. VRAM byte-identical over all six party measurements.

| host object | s1 | s2 | |
|---|---:|---:|---|
| `heap:libc10.so+0x605e5` | 433.0 MB | 1298.8 MB | +865.8 MB |
| `anon:[heap]` | 984.0 MB | 775.3 MB | -208.7 MB |

- 865.8 MB became live host data, the peak rose 87.7 MB: the arena already held 84% of those pages.
- `mpc.py:165:cuda` fell from 866.6 MB to 638.6 MB, not to zero: `Graph.forward` keeps every
  computed value until the pass ends (`_clear_unused_values()` is defined, never called), so device
  values derived from host parameters (transposes, reshapes) stay.
- For a card-fitting goal, -710 MB VRAM for +88 MB host at +4.6% wall is a Pareto point.
