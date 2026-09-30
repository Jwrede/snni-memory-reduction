# s4_ffn_threads: fewer FFN workers on s3

Job 46196361, 55:48, rc=0. Paid candidate on `s3_boot_layer_release` (earlier ordering).

| | |
|---|---|
| reading | at `after_gelu` the pool holds 99,634.1 MiB live, 108,384 reserved (gap 8,749.9 MiB); FFN/GELU runs `nthreads = min(max_threads, 32)` = 8, each with its own stream and temporaries; gap read as per-worker workspace |
| change | 8 -> 2 workers (value from the earlier campaign's `w5`: "-- not the data -- is what reserves the peak") |
| prediction | `after_gelu` +8,416 -> ~+2,100, peak ~-5.8% |

```
                    s3_boot_layer_release   s4_ffn_threads
peak_reserved_kb          110,985,216         114,262,016    +3,276,800 kB   +2.95%
peak_alloc_kb             103,046,739         102,284,499      -762,240 kB   -0.74%
wall                            3,340 s             3,339 s                  -0.03%
gate                         md5 01469da3..., IDENTICAL
```

```
marker              jump s3   jump s4
after_layernorm1     +2,048    +1,760
after_intermediate   +3,552    +3,648
after_gelu           +8,416   +11,808    <- RISES by 3,392 MiB
```

- The reservation is not per-worker workspace: eight workers on eight streams interleave requests that
  reuse blocks; two workers request in a different size order and the pool grows.
- The earlier campaign's tree ran the stage at 32 threads (same expression, larger allocation).
- On the later tree the same change is the published `s12_ffn_threads` (-6.97% on `s9`), declaring
  `refuted_as: s4_ffn_threads`.
