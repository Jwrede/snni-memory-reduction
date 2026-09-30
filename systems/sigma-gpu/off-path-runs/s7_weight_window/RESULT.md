# s7_weight_window: weights resident only per layer

A layer's weights dropped with `madvise(MADV_DONTNEED)` after its forward and rewritten before the
next (the fill is a pure function of a counter).

```
                    peak_host      W       wall   gate
   s6_dealer_window  2,139,588 kB  9.17    63 s   COMM:1062390674 KEYS:18075947008
   s7_weight_window  2,134,240     9.19    63 s   identical
                        -0.25%
```

`SNNI_WW|regions=96|drops=288|refills=192|dropped_bytes=1,359,765,504`

```
   s6_dealer_window                              s7_weight_window
   453.3 MB  0.207  heap:GPUBERT::GPUBERT        -- absent --
   226.6 MB  0.103  heap:main <- bert.h:89       226.6 MB  0.104  (unchanged)
                                                 856.4 MB  0.392  file:key_0.dat
   attributed 1,252 MB, unattributed 938.6       attributed 1,659 MB, unattributed 526.2
```

- `file:key_0.dat` (856.4 MB) is `s6`'s key mapping: largest operation 787 MiB, `dropConsumed()` in
  64 MiB steps, floor 851 MiB; measured 817 MiB.
- The dealer's fill and the online key window are two maxima within 0.3%.
- Build refusal: `host peak 2,134,240 kB is the kernel's VmHWM; the 100 ms poller only observed
  1,621,076 kB (24.0% low)`. Since `s6` the maximum is a spike in the first second.
- Correctness: per-tensor FNV-1a checksum of the original fill, recomputed on all 192 refills, abort on
  mismatch; did not fire.
- First version dropped only after each layer's first run (after `new SIGMAKeygen` pinned its buffer):
  96 drops, 453.3 MB, peak -0.34%.
- `measured_overhead` counted every file-backed mapping as library text, so 817 MiB of key mapping was
  deducted from `W` (first reading 2.02). Fixed 2026-08-19; also raised SHAFT-GPU's `W` (433.3 MB model
  blob) from 13.2-13.4 to 15.6-15.8 at the time.
