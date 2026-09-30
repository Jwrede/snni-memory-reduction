# s4_input_copy_fused: conservation of the reservation

Job 46181712, 55:49, rc=0. Same patch as `../s3_input_copy_fused/` (byte-identical), on
`s3_boot_layer_release` (where `after_bootstrap3`, `after_layernorm2`, `after_bootstrap4` reserve
nothing). Hypothesis: no later phase left for the saving to reappear in.

```
                  s3_boot_layer_release   s4_input_copy_fused
peak_reserved_kb        110,985,216           111,050,752     +65,536 kB   +0.06%
peak_alloc_kb           103,046,739           103,031,891     -14,848 kB   -0.01%
wall                          3,340 s               3,341 s                +0.03%
final reservation         108,384 MiB           108,448 MiB   +64 MiB
```

```
                    jump s3      jump s4     recovered
before_attention    +53,792      +43,104      -10,688   <- the lever
after_attention           0       +2,528       +2,528
after_layernorm1     +2,048       +9,440       +7,392
after_intermediate   +3,552       +4,384         +832
after_gelu           +8,416       +8,416            0   identical
after_final / bootstrap3 / layernorm2 / bootstrap4: 0 in both
                                              -------
                                              +10,752 against 10,688 withheld
```

- Recovery completes before `after_gelu`, two thirds at `after_layernorm1` (not touched by `s3`).
- `before_attention`'s floor is 43,104 MiB (27,648 + 16,128 MiB, two fresh 768-wide arrays).
