# s8_weight_window_early: drop interleaved with the fill

Same lever as `s7_weight_window`, dropping inside the fill loop; plus a `mincore` diagnostic.

```
   s6_dealer_window   2,139,588 kB
   s7_weight_window   2,134,240      -0.25%   drop after the fill, before the key window
   s8_..._early       2,129,116      -0.49%   drop interleaved with the fill
```

```
   SNNI_MHA|wQKV |elems=1769472|bytes=14155776|resident_kb=0|mincore_rc=0|nonzero=0
   SNNI_MHA|wProj|elems= 589824|bytes= 4718592|resident_kb=0|mincore_rc=0|nonzero=0
```

- `heap:main <- bert.h:89` (226.6 MB = 12 x (`wQKV` 14.2 MB + `wProj` 4.7 MB)): zero resident pages,
  never written (`Tensor2D` constructor `data(new T[d1 * d2])`; `_MHADummy` has no `getweights()`).
  The pagemap census reports 226.6 MB the kernel does not hold (README.md).

At `s8`'s peak:

```
   226.6 MB  0.104  heap:main <- bert.h:89        <- zero resident per mincore
   134.2 MB  0.062  heap:SIGMAKeygen              llamaBuf + dummyBuf, measured in s4
   134.2 MB  0.062  heap:SigmaPeer::initCommBufs  measured in s2
   133.0 MB  0.061  heap:Layer::forward
    75.5 MB  0.035  file:nvidiactl                driver
   attributed 802 MB
   unattributed 1,377.9 MB = 66% of the peak
```

Build refusal: `DECOMPOSITION NOT AT THE PEAK` (leader `VmHWM` 2,129,116, poller 1,618,660).
