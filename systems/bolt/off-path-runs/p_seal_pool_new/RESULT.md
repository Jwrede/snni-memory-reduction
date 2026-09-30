# p_seal_pool_new: SEAL MMProfNew on s2

Job 46124468, `bigsmp` r05n01, `OMP_NUM_THREADS=4`, both parties on one node, rc=0; derive 46124469.
Compared with `s2_threads_4` (same binary, same threads).

```
                    peak host        wall     kB saved per added second
s2_threads_4       17,150,684 kB     775 s    --        (n=3, peak spread 0.117%, wall spread 0.383%)
p_seal_pool_new    16,531,708 kB    1515 s    836
                     -618,976 kB    +740 s
                          -3.61%     +95.5%
```

```
s0_default (baseline)   -4.886475   5.286133
p_seal_pool_new         -4.889648   5.284912
                        -3.17e-03  -1.22e-03
```

Gate inside 1.81e-2 and inside `s2_threads_4`'s range (`-4.889404`, `5.287109`).

## Rule 6 comparison

```
                 peak      wall      ratio   kB per added second
s2_threads_4    -2.01%   +15.2%      0.132   3,450
p_seal_pool_new -3.61%   +95.5%      0.0378    836
```

- 3.5x worse than `s2_threads_4` by ratio, 4.1x by kB per second. PUMA's `p_conc1` was refused at 3x.
- `steps/s2_threads_4/levers.env` prose quotes `1.90% / +34.5% = 0.055` (probe run before the step);
  `steps.csv` gives 0.132 and is the authority.
- Freeze-corrected, recorder table (4,100 kB) subtracted: 17,150,684 -> 16,527,608 kB (-623,076 kB,
  -3.63%), 622.4 -> 1240.4 s (freeze 274.6 s, +99.3%), ~1,010 kB/s; `s2_threads_4` over
  `s1_stream_weights`: 351,940 kB for +112.3 s (510.1 to 622.4 s), ~3,130 kB/s.

## Server C calibration

```
                        Server C            PALMA           agreement
base   (s1 @ 4 threads) 17,130,632 kB    17,150,684 kB      0.12%
probe  (pool_new)       16,538,840 kB    16,531,708 kB      0.04%
delta                        -3.45%           -3.61%        0.16 pp

base   wall                     631 s            775 s
probe  wall                     587 s           1515 s
delta                          -7.0%           +95.5%       INVERTED
```

## Instrument share of the wall increase

```
party 3337872   247 samples   mean 655 ms   max 1845 ms   =  162 s
party 3337916   188 samples   mean  35 ms   max   87 ms   =    7 s
                                                             169 s of the 740 s added
```

- ~23% of the added wall is capture time; `MMProfNew` returns each pool on handle death, so the
  weight-holding party's mapping count rises and each scan costs more.
- Remaining ~570 s: not established (hypothesis: page-fault cost of release and re-acquisition on a
  1.46 TB NUMA node).
- Both precedents predicted a rise (earlier campaign 112.6 -> 138-140 GiB; BumbleBee +12.7%); measured
  here: -3.61%.
