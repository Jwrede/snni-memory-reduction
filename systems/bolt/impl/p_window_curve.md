# BOLT: layer-window curve

`bigsmp`, image `bolt-p_window.sif` (md5 `a4c2617472303bbbb2aba615d6f750fc`), one run per k, jobs
46089721 / 46095700 / 46095702 / 46095704 / 46095706; each announced `LEVER|layer_window|k=<n>`.
Control: `p_window_control.md`.

A window of `k` layers holds `k/12` of the weights and keeps `k`-way outer parallelism; `s1` is k=1.

| k | peak kB | peak GiB | wall s | vs k=1 memory | vs k=1 runtime | GiB per % runtime |
|--:|--------:|---------:|-------:|--------------:|---------------:|------------------:|
| 1 | 17,496,324 | 16.69 | 685 | -- | -- | -- |
| 2 | 24,690,876 | 23.55 | 705 | +6.9 | +2.9% | 2.3 |
| 3 | 33,652,920 | 32.09 | 674 | +15.4 | -1.6% | 9.6 |
| 4 | 42,811,652 | 40.83 | 655 | +24.1 | -4.4% | 5.5 |
| 6 | 60,173,444 | 57.39 | 615 | +40.7 | -10.2% | 4.0 |

- Memory grows ~8.2 GiB per resident layer.
- k=1 saves 42.7 GiB against k=6 for 11.4% more run time: k=1 stands (`s1` already is k=1).
- Run-time spread known for k=1 only (control: wall 2.50%, peak 0.59%). k=2, 3 lie inside it; k=4, 6
  outside and monotone (~10% at six layers).
