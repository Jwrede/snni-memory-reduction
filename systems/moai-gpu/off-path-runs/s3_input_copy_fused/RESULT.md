# s3_input_copy_fused: at its arithmetic floor, below the metric's resolution

Job 46165673, 55:59, rc=0. Derived from pool growth events.

| | |
|---|---|
| change | shrink element i before copying element i+1 (stock: copy the whole 768-wide input at full level, then mod-switch) |

```
                         p_finalprobe (s2 + markers)   s3_input_copy_fused
keys_created                       40,608 MiB                40,576 MiB
before_attention                   94,400                    83,712      jump 53,792 -> 43,136
after_attention                    94,400                    86,400
pool_used at before_attention      68,146.1                  68,146.1    IDENTICAL
```

- Floor: `enc_ecd_x` stays fresh (768 x 36 MiB) beside the shrunk copy (768 x 21 MiB):

```
27,648 + 16,128 = 43,776 MiB floor       measured 43,136
```

- The patch header predicted ~-26,000 MiB (compared against what the phase holds at its end, not what
  it demands at once).

```
peak_reserved_kb   s2 123,928,576   ->   this run 124,747,776    +819,200 kB = +0.66%
peak_alloc_kb      s2 108,321,427   ->   this run 108,321,427    byte-identical
wall               s2 3,342 s       ->   this run 3,350 s        +0.24%
```

- Against `p_finalprobe` (s2 plus markers) the same run is 1,408 MiB lower (-1.14%). Unresolved at n=1.
- Reserved-peak spread over four runs: 3,520 MiB (2.94%); `s2` and `s3_inter_output_free` have
  identical device tables and differ by 1,312 MiB (1.08%) reserved.
- `peak_alloc_kb` 108,321,427 in four runs across four images: insensitive.
- Co-tenancy: shared node r18n02 with `p_finalprobe`'s recorder channel for ~10 minutes (another GPU);
  `run_step.sh` now chains a later invocation behind the last measurement job per system.
