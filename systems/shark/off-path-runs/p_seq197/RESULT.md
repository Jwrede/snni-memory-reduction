# p_seq197: sequence length 197

Diagnostic (different operating point; README.md, Transfer lines and sequence length).
`SHARK_SEQ_LEN=197` on the published `s0_default` and `s5_up_split` images, 3 replicates each,
nothing rebuilt. SHARK loads no model, so this is a geometry change.

## Maximum over parties (current convention, from `party_peaks.txt`)

| step | runs (kB) | party | median |
|---|---|---|---:|
| `s0_default` | 100,740,244 / 100,740,196 / 100,736,424 | 0 | 100,740,196 |
| `s5_up_split` | 541,304 / 540,056 / 538,284 | 1 | 540,056 |

Factor 186.54x at 197 tokens. At 128 tokens the published line is 123.65x.

## Party 0 only (first version)

```
                          run1          run2          run3        median        rc
s0_default          100,740,244   100,740,196   100,736,424   100,740,196     7
s5_up_split             529,376       529,480       529,020       529,376     0
```

Raw `VmHWM`; the recorder table is under 1 MB here.

## `rc=7` on the baseline

Both parties exit 0, the gate is complete, the dealer finished. The guard reports
`produced no LEVER| marker`: `s0_default` has no lever. Not fixed (shared code).

## `VmHWM` against the poller maximum

```
             VmHWM        poller max     apart
s0 run1  100,740,244    100,671,592     68,652 kB   0.068%
s0 run2  100,740,196    100,345,804    394,392 kB   0.39%
s0 run3  100,736,424    100,358,388    378,036 kB   0.38%
s5 all       identical to the kilobyte in all three runs
```

```
VmHWM  spread over three runs   3,820 kB  = 0.0038%
poller spread over three runs 325,788 kB  = 0.32%
```

The baseline peak is a transient the 100 ms poller catches at different points; at `s5` the peak
is a plateau.

## Why the factor grows

`s3_key_stream` removes the dealer's key material, which scales with the token count; what remains
after `s5` does not.
