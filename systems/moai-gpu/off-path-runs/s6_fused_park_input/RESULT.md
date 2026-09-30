# s6_fused_park_input: LN1 residual parked per element

Job 46192748, 55:49, rc=0. Third attempt on `before_attention` (earlier ordering). The LN1 residual is
copied, mod-switched and parked one element at a time (never a device array).

```
                    s3        s5        s6
before_attention   94,368    94,400    67,552
jump              +53,792   +53,824   +26,976     <- -26,816 MiB
```

- `LEVER|fused_park_input|parked_ciphertexts=768|disk=1`, bank 16,911,437,824 B.
- Floor: `enc_ecd_x` 27,648 MiB plus one element in flight; measured 26,976.

```
                  s3_boot_layer_release   s6_fused_park_input
peak_reserved_kb        110,985,216            110,919,680    -65,536 kB   -0.06%
peak_alloc_kb           103,046,739            103,044,179     -2,560 kB   -0.00%
wall                          3,340 s                3,340 s               unchanged
gate                            md5 01469da3..., IDENTICAL
SPILL_BYTES                              16,911,437,824
```

Later phases reserve 26,752 MiB more against 26,816 saved.

```
step                                   where it acts                 result
s2_boot_input_release   bootstrap loops, after after_gelu            -9.0%   WORKED
s3_boot_layer_release   boot_layer -> layernorm2 and bootstrap3      -10.4%  WORKED
s4_input_copy_fused     before_attention, re-ordered                  null
s5_park_input_copy      before_attention, parked after the loop       null
s6_fused_park_input     before_attention, never built at all          null
```

Rule: a reduction lowers the peak only at or after the last growth event.

```
keys_created       +40,576 MiB   37.4%   permanent by construction, not a target
before_attention   +26,976       24.9%   at its floor, and refilled anyway
after_intermediate +12,448       11.5%   between the two, untested
after_gelu          +8,544        7.9%   the LAST growth event, and therefore the live target
```
