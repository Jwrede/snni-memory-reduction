# Scouting note: p_seal_pool_new

Scouting run, not evidence: `jwmaster01` (Server C), Ubuntu 24.04, 16 cores, not exclusive, no
freeze, no poller; `maxRSS` from `/usr/bin/time -v`; one run per arm. Decided by PALMA job 46124468.

Binaries extracted from their PALMA images, md5 checked (`18a9f64f` `s1_stream_weights`, `914e31ea`
`p_seal_pool_new`), `OMP_NUM_THREADS=4`.

```
s1_stream_weights   maxRSS 17,130,632 kB   wall 631 s   lever markers 0
p_seal_pool_new     maxRSS 16,538,840 kB   wall 587 s   lever markers 2
                          -591,792 kB            -44 s
                              -3.45%             -7.0%
```

```
s1_stream_weights   1,-4.88623,5.29077
p_seal_pool_new     1,-4.89355,5.28149
```

Score differences 0.0073 and 0.0093, inside the replicates' own spread:

```
s2 run1/2/3   5.287109  5.278564  5.289306     spread 0.01074
s1 run1/2/3  -4.886230 -4.890137 -4.888428     spread 0.00659
```

Precedents predicted a rise: earlier campaign `MMProfNew` on BOLT 112.6 -> 138-140 GiB; BumbleBee
`x4_seal_pool_new` +12.7%.

## Outcome (2026-08-14, job 46124468)

```
                     this note (Server C)      PALMA
peak delta                  -3.45%            -3.61%     agree to 0.16 pp
wall delta                  -7.0%             +95.5%     SIGN INVERTED
```

Absolute peaks agree to 0.12% (base) and 0.04% (probe). Server C is used to scout memory, not run
time. Full account: `off-path-runs/p_seal_pool_new/RESULT.md`.
