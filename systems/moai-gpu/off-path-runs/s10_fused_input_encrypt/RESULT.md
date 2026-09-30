# s10_fused_input_encrypt: input-encrypt fusion on s9's tree

Retry of `s7_fused_input_encrypt` (refuted on `s6`'s tree) after `s7_stream_local_reload` had removed
four later growth events.

```
                     s9_stream_boot4    s10          predicted
before_attention  reserved   67,552    52,576       52,576      exact
after_attention   growth     +2,496   +17,920                   the refill
peak_reserved_kb      80,510,976   80,084,992                   -0.53%
peak_alloc_kb         69,072,019   67,749,011                   -1.92%
wall                       3,426        3,408 s
gate                 md5 01469da3..., BYTE-IDENTICAL for the tenth consecutive run
```

```
                     before_attention    after_attention
                     reserved   live     reserved   live
s9_stream_boot4        67,552  52,018      70,048  53,554
s10_fused_input        52,576  52,018      70,496  53,554
```

- The attention block ends at ~70,000 MiB reserved either way, with 53,554 MiB live: ~16,900 MiB of
  reservation above its live set, independent of the pool's state on entry.

Refutations of the same shape on this system:

```
s3_input_copy_fused    largest growth event cut to its arithmetic floor          null
s5_park_input_chunk_ffn that event halved, 53,792 -> 26,976 MiB                 -2.30%
s7_fused_input_encrypt 14,944 MiB removed, 32 MiB off the forecast              +0.04%
s8_stream_boot1_ln1    9,504 MiB removed at LayerNorm1                          -1.06%
this step              14,976 MiB removed at input preparation                  -0.53%
```

Worked: `s6_stream_boot23` (-22.42%), `s9_stream_boot4` (-2.19%), both removing live data from the
phase holding the maximum. The device table has no row for the 16,900 MiB (rows: the key bank, and
`encrypt_zero_asymmetric_internal`, the 768 input ciphertexts). Next: an allocator trace.
