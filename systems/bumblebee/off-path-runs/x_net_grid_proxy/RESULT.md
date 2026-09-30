# x_net_grid_proxy: OT-instance reduction over a bandwidth x latency grid

Diagnostic. Question: how much of the paid phase's (`b5`-`b8`) run-time cost depends on the loopback
link; peak recorded as well.

## Setup

| | |
|---|---|
| image | `bumblebee-f_b4_faithful.sif` (published `b4`), 12 layers, `s=128`, 16 threads per party |
| configurations | `BB_MAX_CONCURRENCY` 16 (`b4`) or 1 (`b8`); nothing rebuilt |
| machine | PALMA `normal`, 36-core Skylake, 95 GB, exclusive, both parties on one node |
| runs | one per cell |
| peak | each party's `VmHWM`, maximum over parties; no recorder (line references with recorder: `b4` 356.9 s, `b8` 1,335.0 s) |

## Userspace shaper

PALMA sets `user.max_net_namespaces=0` (`unshare -rn` fails with `ENOSPC`), so the earlier `tc tbf` +
`netem` chain on a private `lo` cannot be built. Tools in `tools/`:

| tool | function |
|---|---|
| `connshim.c` (`connshim.so`, `LD_PRELOAD`) | rewrites `connect()` to ports 9530/9531 into 19530/19531; BumbleBee unpatched |
| `shaper.py` | forwards each chunk after RTT/2; one token bucket for all traffic; per-direction queue max(4 MiB, rate x 0.6 s) |
| `rttprobe_proxy.py` | measures the round trip before each run (`cells/*/rtt.txt`); every cell within 1.2 ms of nominal (0.08-0.09 ms at RTT 0) |

Validation on the earlier grid's image (`val/`):

```
cell                   shaper     netem     difference   peak difference
bw1000 rtt0  c16        365 s     347 s       +5.2 %        <= 1.5 %
bw50   rtt0  c16      1,947 s   1,960 s       -0.7 %
bw1000 rtt80 c16      1,551 s   1,533 s       +1.2 %
bw200  rtt40 c1       2,211 s   2,177 s       +1.6 %
```

## Result

`bb_wall_s`, one instance / 16 instances, ratio:

```
                 RTT 0 ms              RTT 20 ms             RTT 40 ms             RTT 80 ms
1000 Mbit/s    981 /  243  4.04    1439 /  707  2.04    1885 / 1144  1.65    2755 / 2025  1.36
 200 Mbit/s   1146 /  417  2.75    1578 /  849  1.86    2008 / 1279  1.57    2857 / 2142  1.33
  50 Mbit/s   1856 / 1190  1.56    2225 / 1543  1.44    2620 / 1919  1.37    3447 / 2735  1.26
```

- All 24 runs exit 0 on both parties with all 98,304 gate values.
- On slower links the 16 instances wait on the link, so removing them costs less.
- Line comparison: `b8` / `b4` = 1,335.0 / 356.9 = 3.7x (with recorder, raw loopback).

Peak, maximum over parties:

```
16 instances   6,730,568 - 6,848,932 kB   6.89 - 7.01 GB   (line b4: 6,913 MB)
 1 instance    1,453,100 - 1,556,472 kB   1.49 - 1.59 GB   (line b8: 1,496 MB)
```

| | 1 Gbit/s | 200 Mbit/s | 50 Mbit/s |
|---|---:|---:|---:|
| one instance, mean over RTTs | 1,466,372 kB | 1,502,176 kB | 1,542,576 kB (+5.2%) |
| 16 instances | 6,753,116 kB | | 6,828,158 kB (+1.1%) |

No trend with RTT. Traffic per run: 6,776,159,794 to 6,776,238,520 bytes (16 instances),
6,726,090,489 to 6,726,149,332 (one instance), constant to 0.002%.

## Files

`cells/<cell>/`: `CELL_META.txt`, `RESULT.txt`, `rtt.txt`, `comm.txt`, `shaper_stats.json`,
`binary_md5.txt`. `val/`: four validation cells and a one-layer smoke run. Figure script:
`master-thesis/figures/ch5_rework/make_bb_net_sweep.py`.
