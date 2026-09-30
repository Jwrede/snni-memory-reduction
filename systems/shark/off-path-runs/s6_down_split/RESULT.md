# s6_down_split at T=16

Runs `results/s6_down_split/run1-3` (jobs 47061298-300, derives 47061301-03), 2026-09-30, EPYC 7742,
16 threads. Program: `s5_up_split` plus the FFN down projection as four 768-wide row chunks, each
with its own 25.2 MB triple (`impl/src/bert_instrumented_s6_down_split.cpp`). `levers.env` quotes
the T=4 numbers.

| | s5_up_split | s6_down_split | change |
|---|---:|---:|---:|
| peak_host (median replicate, max over parties, table subtracted) | 469,792 kB | 469,944 kB | +0.03% |
| replicate spread | 0.282% | 0.126% | |
| wall, median corrected | 95.0 s | 96.0 s | +1.1% |
| `recv_array<unsigned __int128>` in the published party's table | 91.3 MB | absent | |
| `anon:[heap]` in the published party's table | 161.5 MB | 296.8 MB | |

- The freed Beaver spans stay in the allocator. Not a step; the line ends at `s5`.
- Published party: leader at `s5`, the other online party at `s6` (peaks within 1 MB).
- T=4 measurement: `s6_down_split_t4/`.
