# p_phasetrigger: joining the peak instant to a named phase

Specification, 2026-08-08. Not implemented (touches shared code).

## Problem

The census triggers on an RSS threshold (95% of the step's clean peak), not on a phase:

```
a1   trigger 307,560,716 kB   161.2 GB live, 153.8 GB retained
a2   trigger 241,033,452 kB   179.9 GB live,  66.9 GB retained
```

`rlwe.NewElement` rose 107,783.2 -> 120,680.4 MB across the step (24,598 objects of 524,288 B); the
instrument cannot tell a larger working set from a differently placed snapshot.

## Rejected

Piping the program's stdout through a timestamper (changes the measured process's output path and
buffering, needs a FIFO).

## Proposal

`poll_peak.sh` (samples `VmHWM` every 15 ms, writes on each new high-water mark) gets two optional
variables:

| variable | meaning |
|---|---|
| `SNNI_POLL_PHASE_FILE=<path>` | the program's own stdout file |
| `SNNI_POLL_PHASE_RE=<regex>` | lines that count as phase boundaries |

On each new HWM the poller records the last matching line (read-only):

```
<timestamp_ms> <VmHWM_kB> <last phase line>
```

Unset by default, byte-identical behaviour without them. Pending a decision on changes to the shared
instrument. Records where the peak was, not why.
