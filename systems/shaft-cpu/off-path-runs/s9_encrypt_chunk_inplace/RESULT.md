# s9_encrypt_chunk_inplace

On the `s8` runs before the 2026-09-29 re-measurement. Off-path.

| | |
|---|---|
| change | `crypten.cat(enc_chunks, dim=0)` replaced by writes into a preallocated output |
| marker | `LEVER|encrypt_chunk_inplace_cpu|patched|chunks=8` |
| peak | 898,152 -> 891,612 kB, -0.7%, against `s8`'s 8.80% spread |
| gate | byte-identical (`-1.0620574951e+00 8.0404663086e-01`) |
| transcript | identical (bytes=11,228,688,384, rounds=1525) |

```
                         replicates (kB)                  median     vs s8
s8_encrypt_chunk    924,844 / 898,152 / 845,840          898,152       --
s9_encrypt_chunk_inplace  891,612 / 881,416 / 903,452    891,612     -0.7%
```

## Effect at the site (poller, aligned on `embed|after_convert`)

```
s8   +1,334 ms  rss=638,808  hwm=647,644
s8   +1,624 ms  rss=317,500  hwm=773,720      <- +126,076 kB
s9   +1,285 ms  rss=590,492  hwm=650,580
s9   +1,500 ms  rss=563,520  hwm=656,256      <-   +5,676 kB
```

## Prediction and error

- Predicted -20% to -25%: embedding share 28,996 x 768 x 8 B = 173,976 kB, `crypten.cat` holds
  inputs and output, 2 x 173,976 = 347,952 kB against a 384,348 kB "transient".
- 384,348 kB was poller maximum minus the nearest phase marker, a difference over many
  allocations rather than one allocation.

High-water trace of `s9`: 29 increases above 3 MB, 616,716 kB in total, from `main|after_init` to
`layer|before_run_7`:

```
+15,488 -> 306,244   during model loading
+27,780 -> 338,964
+29,460 -> 368,424
+17,244 -> 416,720
+35,232 -> 462,312
+10,260 -> 471,664   during input encryption
+94,972 -> 565,564   <- largest single event, 11% of the peak
+83,648 -> 649,212
+18,432 -> 682,404   during layer 0
+12,292 -> 706,900
+12,288 -> 719,884
+24,576 -> 744,460
+49,976 -> 794,436
+47,692 -> 849,528   during layer 1
 +3,068 -> 853,364
+23,592 -> 872,120   during layer 2
+44,192 -> 916,312   during layer 7   <- last increase
```

## `p_graphop_s8` (`graphop_markers_cpu`, per-node VmRSS and VmHWM)

```
776,248 kB   already at the first graph node
835,588 kB   +59,340  after layer.intermediate.intermediate_act_fn.GELU
845,332 kB    +9,744  after the same GELU in another layer
```

- 91.8% of the peak is set before any graph node runs.
- Graph execution adds 69,084 kB (8.2%), all from the GELU; the high-water changes 3 times over 15
  subgraphs.
- 71.5% of `s8`'s peak is `anon:residual`.
- The only GELU lever (earlier campaign's `04_reduce_temp`, polynomial GELU) is an approximation
  choice; upper bound of its effect here: 69,084 kB.

## Line at the time

```
s0_default          3,575,304 kB   W = 19.80
s7_embed_chunk      1,080,140
s8_encrypt_chunk      881,272      W = 4.31
                                   3,575,304 -> 881,272 = -75.4%, factor 4.057x
```

Superseded by the 2026-09-29 re-measurement (`steps.csv`).
