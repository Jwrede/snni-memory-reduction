# p_depth3: third stack frame for the device recorder

Job 46152636, `gpuh200`, 56:34, on `s2_boot_input_release`'s image. Probe (site keying only).

At depth 2 the two largest rows had no owner outside Phantom:

```
42316.3 MB  33.3%  encrypt_zero_symmetric        <- make_cuda_auto_ptr
33843.8 MB  26.7%  mod_switch_scale_to_next      <- make_cuda_auto_ptr
```

| change | |
|---|---|
| `cudarec.c` | optional third frame (`caller_frame2()` returns `bt[4]` at depth 3); site key becomes the triple; sixth table column; depth 1 and 2 unchanged (selftest: `at depth 2 the three owners of one helper are three distinct sites`) |
| defect in the first attempt | `SNNI_CUDAREC_DEPTH` parsed as `d[0] == '2'`; `3` gave depth 1 (`# depth=1` in the header); one 56-minute run lost. The parser now reads the number, range-checks 1..3, aborts otherwise |

```
42278.6 MB  generate_one_kswitch_key <- encrypt_zero_symmetric   <- make_cuda_auto_ptr
33843.8 MB  rescale_to_next          <- mod_switch_scale_to_next <- make_cuda_auto_ptr
16911.4 MB  libc.so.6+0x29d90        <- test.cu:70:main          <- single_layer_test():868
 8053.1 MB  mod_switch_to_next_inplace <- mod_switch_to_next     <- make_cuda_auto_ptr
 6442.5 MB  single_layer_test():1073 <- gelu_v2                  <- operator=
```

- Rank 1: the key-switching keys (`generate_one_kswitch_key`), matching the key bank (~40,325 MiB
  from source); live for the whole run.
- Rank 2: per-rescale intermediates created and destroyed inside Phantom; no MOAI owner.
- Only rank 5 leads with a MOAI frame (GELU loop, dispositioned as `s3_gelu_res_move`).
- More depth does not reach MOAI code. Next candidates: long-lived MOAI arrays (`boot_layer`, residual
  banks, `enc_X`), named in this table by their owning frames.
