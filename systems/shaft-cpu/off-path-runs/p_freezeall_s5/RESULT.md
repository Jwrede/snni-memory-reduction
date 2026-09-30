# p_freezeall_s5: `s5_per_layer` with every poller freezing both parties

Opt-in instrument variant, 2026-09-29, 3 replicates, not published. Compared with the `s5` runs of
2026-08-18 (jobs 46242085-87), which the 2026-09-29 re-measurement (jobs 47060374-76) replaced.

## Purpose

In the 2026-08-18 runs the published peak was run3's non-leader peak (1,715,256 kB raw), whose table
was sampled at 1,530,192 kB, 0.892 of that peak (limit 10%). The row decomposed with the leader's
table (`decomposed_party = leader`, `peak_party = non-leader`). Goal: an aligned, atomic table for
the heavier party.

## Instrument change (reverted)

| | |
|---|---|
| flag | `SNNI_PM_FREEZE_ALL_POLLERS=1` |
| effect | every party's poller sends `SIGSTOP` to all parties at its own party's new maxima, captures, sends `SIGCONT`; `flock` on `<run>/freeze.lock` serialises the windows |
| `frozen_ms` | sum over pollers (`peaks/PAUSE_INDEX`); RUN_META `freeze_all_pollers=1` |
| files touched | `tools/palma/conf/shaft-cpu.conf`, `tools/palma/measure.sh` (`snni_run_meta`, `snni_start_poller`), `tools/poll_peak.sh` |
| patch | `freeze_all_pollers.patch` here; `git apply --check` clean against the instrument of 2026-09-29 |

## Setup

| | 2026-08-18 (46242085-87) | this run (47049918-20) |
|---|---|---|
| partition | `express`, `--exclusive`, `--exclude=r07n04` | same |
| node | r05n10 | r07n01 (both 95 GB `express`) |
| image | `shaft-cpu-s0_default.sif`, digest `f35e71f6...a3abfe` | same |
| `levers.env` md5 | `4609f18c5783a6b219f664d8fef1f76f` | same |
| `script_md5` / `runner_md5` | `f67d5779...2ac51281` / `37fd3e15...fe838f73` | same |
| levers | `plaintext_free`, `triple_share_free`, `trace_nograd`, `onnx_single_file_export`, `encrypt_spill` | same |
| `omp_num_threads` | 8 | 8 |
| `pm_min`, `fullscan`, `freeze`, `poll_ms`, `pm_depth` | 65536, 1, 1, 15, 1 | same |
| `freeze_all_pollers` | absent | 1 |

Derives: leader tables jobs 47049921-23 (`derive.sbatch`), non-leader tables 47050204-06
(`tools/derive_party.sbatch`, `logs/run*/derive_party-<job>.out`).

## Results

Per-run peaks, `VmHWM` raw kB:

| run | leader | non-leader | replicate peak | freezes (leader / non-leader) | frozen ms | wall s | corrected wall s |
|---|---:|---:|---:|---|---:|---:|---:|
| 08-18 run1 | 1,701,440 | 1,715,780 | 1,715,780 | 116 / 0 | 6,213 | 106 | 99.8 |
| 08-18 run2 | 1,705,592 | 1,592,032 | 1,705,592 | 141 / 0 | 7,166 | 106 | 98.8 |
| 08-18 run3 | 1,585,300 | 1,715,256 | 1,715,256 | 117 / 0 | 6,060 | 107 | 100.9 |
| run1 | 1,606,392 | 1,604,908 | 1,606,392 | 674 / 684 | 63,361 | 166 | 102.6 |
| run2 | 1,715,220 | 1,717,452 | 1,717,452 | 839 / 848 | 77,171 | 180 | 102.8 |
| run3 | 1,605,508 | 1,605,284 | 1,605,508 | 679 / 679 | 62,317 | 165 | 102.7 |

`make_steps.py` row from each set (scratch copy; `steps.csv` unchanged):

| field | 08-18 runs | these runs |
|---|---:|---:|
| `peak_host` (median, instrument subtracted) | 1,711,156 | 1,590,008 |
| instrument subtracted | 4,100 | 16,384 |
| `peak_host_min` / `peak_host_max` | 1,705,592 / 1,715,780 | 1,605,508 / 1,717,452 |
| `spread_host_pct` | 0.594 | 6.969 |
| `peak_party` / `decomposed_party` | non-leader / leader | leader / leader |
| `frozen_ms` (median run) | 6,060 | 63,361 (38.2% of wall) |
| corrected wall, median | 99.8 s | 102.7 s |
| `runtime_delta_pct` / `cost_type` | +1.7 / free | +4.7 / paid |

- Gate identical in all six runs (`-1.0504302979e+00 8.0657958984e-01`); transcript identical
  (11,228,688,384 bytes, 1,495 rounds).
- All six tables at `snapshot_frac_of_peak = 1.0`.

## Interpretation

- The extra freezes change the timing that sets the peak. In the 08-18 runs the two parties peaked
  independently at ~1.59 or ~1.71 GB, and the maximum over parties was the upper level in 3/3 runs.
  Here both parties land on the same level per run (differences 224-2,232 kB), the lower one in 2/3
  runs. Median -121,148 kB, spread 0.594% -> 6.969%, with program, levers and image unchanged.
- The recorder (`pmrec.so`, `pmsample`) was rebuilt on PALMA on 2026-08-23; its table is 16,384 kB
  here against 4,100 kB in the 08-18 runs.

## Decision

Not published (Jonathan, 2026-09-29). Instrument change reverted.

## Contents

- `freeze_all_pollers.patch`
- `logs/run{1,2,3}/`: RUN_META, `gate.txt`, `wall_seconds.txt`, `PEAK.txt`, markers, both pollers'
  traces (`poll_<pid>/`: `peaks/INDEX`, `PAUSE_INDEX`, `INSTRUMENT_KB`, `peak.smaps`),
  `freeze.lock`, leader and non-leader tables (`objects_{host,resident,pycensus}[_<pid>].jsonl`).
  Intermediate `*.smaps` removed per `.gitignore`. Per-party peaks: `hwm_kb` lines in
  `derive_party-<job>.out`.
