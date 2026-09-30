# s7_stream_local_reload: FFN reload on the worker's stream

`patch_stream_local_reload.py`, one line. Measured on `s6_stream_boot23`'s tree; not a published row.

| | |
|---|---|
| target | reservation gap, not an object: maximum live set (after_bootstrap4) 64,306 MiB against a peak reservation of 82,144 MiB (17,838 MiB, 21.7%) |
| donor | earlier campaign's `s24_stream_local_reload` (65.19 -> 60.97 GiB): "the per-ct default-stream syncs were the fragmentation source -- with them gone, the reserved landscape is flat at 61760 across boot2/FFN/GELU" |
| change | the GELU loop (`#pragma omp parallel num_threads(nthreads)`, `auto &stream = stream_pool[tid]`) already passes the thread stream to `gelu_v2`; the reload now gets it too. `reload_cipher_from_host` ends in `cudaStreamSynchronize` on the given stream: on the default stream every worker synchronised for each of 3,072 ciphertexts |
| untouched | the two serial reload sites (no thread stream in scope; the patch checks them) |

```
peak_reserved_kb  84,115,456 -> 83,197,952   -917,504 kB   -1.09%
peak_alloc_kb     77,650,579 -> 77,650,579   identical to the byte
wall                   3,399 ->      3,409 s               +0.29%
host peak          2,173,168 ->  2,171,044 kB              -0.10%
gate              md5 01469da3..., byte-identical for the seventh consecutive run
```

-1.09% is below the 2.94% spread.

```
growth event          s6      s7
keys_created      +40,576  +40,576
before_attention  +26,976  +26,976
after_attention    +2,624   +2,432
after_layernorm1   +9,600   +9,504
after_bootstrap2       +0       +0
after_intermediate     +0       +0
after_gelu         +1,344       +0   <- gone
after_final            +0       +0
after_bootstrap3     +864       +0   <- gone
after_layernorm2     +160       +0   <- gone
after_bootstrap4       +0   +1,760   <- new
------------------------------------
peak               82,144   81,248
```

- Reservation flat at 79,488 MiB across FFN and GELU; 1,760 of the 2,368 MiB removed reappear at
  `after_bootstrap4`.
- Consequences: `stream_boot4` becomes applicable (bootstrap 4 is now the last growth event);
  `after_layernorm1` stands out (used 58,930, reserved 79,488, +9,504 for +768 MiB live): target of
  `s8_stream_boot1_ln1`, which is built on this tree and carries the stream fix.
- Free by mechanism (only the stream a copy is queued on changes). Not a row: a free step after the
  paid `s5_park_input_chunk_ffn`, `s6_stream_boot23` (rule 6). The reload call it changes exists only
  since `s5` put the FFN output on the host.
