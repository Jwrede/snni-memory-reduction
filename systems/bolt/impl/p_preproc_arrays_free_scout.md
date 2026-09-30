# Scouting note: s3_preproc_arrays_free

Scouting run, not evidence: `jwmaster01` (Server C), 16 cores, not exclusive, no freeze, no poller;
peak is `Maximum resident set size` from `/usr/bin/time -v`. Decided by PALMA job 46134213.

| | |
|---|---|
| change | free four per-row arrays that `Linear::params_preprocessing_ct_pt` allocates with `new[]` and never releases: 1019 MB over 12 layers and 3 call sites |
| prediction | half written, half never touched: ~510 MB resident; ~1019 MB if glibc made the untouched half resident |

```
                          maxRSS party1      wall     lever
s1_stream_weights (base)  17,124,716 kB      551 s      0     run A, 20:12
s1_stream_weights (base)  17,200,760 kB      561 s      0     run B, 19:47
s3_preproc_arrays_free    16,212,604 kB      559 s      1
```

```
against base A   -912,112 kB   -5.33%
against base B   -988,156 kB   -5.74%
predicted, written half only    497,559 kB   (-2.90%)
predicted, both halves         995,117 kB   (-5.81%)
```

- Base-run spread on this machine: 0.44%.
- Result matches both halves: the untouched `matrix1`/`matrix2` (3 kB, 12 kB) come from the glibc
  arena, where neighbouring allocations dirty the same pages.

```
s1_stream_weights   1,-4.89014,5.28687
s3_preproc_arrays_free   1,-4.89038,5.28711
```

## Harness defect (four failed attempts)

`error: bind: Address already in use`; party 2 then waited ~20 minutes.

```
party 2 connections -> 127.0.0.1:47151, 47178, 47193, 47259, 47304
party 2 own client ports   46302, 39908, 44268, 56502, 37468
/proc/sys/net/ipv4/ip_local_port_range   32768  60999
```

- Both parties on one machine, ~155 ports bound (one per channel); the base port lay inside the
  ephemeral range, so party 2's client sockets competed with party 1's binds. The original harness
  (33000-33400) was in the range too.
- Fix: base port 22000. The liveness check used `kill -0 $P0` (true for an unreaped child); it now
  reads party 1's log.
