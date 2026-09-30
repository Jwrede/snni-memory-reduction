# s4_inter_output_free: reuse levers compete

Job 46185883, 55:34, rc=0. Same patch as `../s3_inter_output_free/`, on `s3_boot_layer_release`.

Forecast from existing series:

```
marker             jump s2    jump s3_inter
after_gelu          +8,640         +1,344    <- the lever cuts it by 7,296
after_bootstrap3         0         +4,224    <- taken back
after_layernorm2   +12,544        +14,400    <- taken back (+1,856)
```

Both reclaiming phases reserve nothing after `s3_boot_layer_release`: forecast up to -7,296 MiB (~-6.7%).

```
                     s3_boot_layer_release   s4_inter_output_free
peak_reserved_kb           110,985,216            108,363,776    -2,621,440 kB   -2.36%
peak_alloc_kb              103,046,739             96,775,763    -6,270,976 kB   -6.09%
wall                             3,340 s                3,325 s                  -0.45%
gate                               md5 01469da3..., IDENTICAL
```

```
marker              s3        s4      jump s3   jump s4
after_gelu       108,384   100,992     +8,416    +1,216   <- -7,200, as predicted
after_final      108,384   100,992
after_bootstrap3 108,384   105,824          0    +4,832   <- was ZERO in s3
after_layernorm2 108,384   105,824          0         0
```

- `s3_boot_layer_release` had fed `boot_layer`'s 768 freed blocks to both `layernorm2` and
  `bootstrap3`; releasing `inter_output` earlier lets GELU take them, so `bootstrap3` reserves again.
- -2.36% is inside the 2.94% spread (pure-noise floor 1.08%); not decidable at n=1. Open: release only
  the elements bootstrap 3 does not need, or order the releases.
