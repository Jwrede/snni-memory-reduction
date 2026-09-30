# p_finalprobe: the published device peak is a cumulative reservation

Job 46164147, 56:03, rc=0. Probe: markers only. Full series in `series.tsv`.

Question: `s3_inter_output_free` measured `peak_alloc_kb` 105,783.6 MiB, above every marker; the only
unmarked phase between `after_gelu` (99,634.1) and `after_final` (72,754.1) is
`ct_pt_matrix_mul_wo_pre_w_mask_single`.

```
final_entry   pool_used 71,986.1 MiB    pool_reserved 108,384
final_i0                71,991.2                      108,384
final_i8                72,039.2   +48.0              108,384
...  every 8 iterations, +48.0 MiB, no exception, sixteen points ...
final_i120              72,711.2                      108,384
after_final             72,754.1                      108,384
```

- 6.0 MiB per iteration, 725 MiB over the loop; `pool_reserved` unchanged. The peak is not there.
- `final_entry` 71,986.1 = `after_gelu` 99,634.1 minus the 27,648.0 released by `swap(inter_output)`.

## The published number

- Final `pool_reserved` 123,232 MiB = `peak_reserved_kb` 126,189,568 kB exactly: the pool is monotone;
  the published peak is everything Phantom ever reserved.
- `peak_alloc_kb` 108,321,427 in four runs across four images (`s2_boot_input_release`,
  `s3_inter_output_free`, this probe, `s3_input_copy_fused`).

```
phase                 used       reserved    ratio
-> before_attention  +27,648     +53,792      1.95
-> after_layernorm1     +768      +1,408      1.83
-> after_intermediate +27,648     +4,960      0.18
-> after_gelu         +6,144      +7,616      1.24
-> after_bootstrap3  +15,360        +352      0.02
-> after_layernorm2     +768     +14,496     18.90
```

- `keys_created` reserves 40,608 MiB first (key bank).
- `after_bootstrap3`: 15,360 MiB live for 352 MiB reserved (reuse after `s2`). `after_layernorm2`:
  14,496 MiB reserved (11.8% of the peak) for 768 MiB live.
- Recorder channel cancelled mid-run (to protect a concurrent step's wall time); no host table.
