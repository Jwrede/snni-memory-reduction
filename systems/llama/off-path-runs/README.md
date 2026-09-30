# LLAMA online: diagnostic runs (2026-08-04)

Six runs on one diagnostic image (`llama-online-p_iotrace.sif`): the measured tree plus per-socket
`want`/`got`/`errno` tracing on both parties and a runtime depth parameter. Not measurements; their
peaks are not quotable (wrong depth in five of six, degenerate gate in all six).

Question: why the baseline died on `always_assert(bytes == recv(...))` in `SocketBuf::read`.

| run | depth | key material | freeze | outcome |
|---|---:|---:|---|---|
| `p_iotrace` | 1 | 1.89 GiB | on | completes, 1303 operations, both traces identical operation for operation |
| `p_iotrace_L2` | 2 | 3.74 GiB | on | completes, 2569 operations |
| `p_iotrace_L4` | 4 | 7.45 GiB | on | fails at operation 20 |
| `p_iotrace_L8` | 8 | 14.86 GiB | on | fails |
| `p_iotrace_L12` | 12 | 22.27 GiB | on | fails at operation 2 |
| `p_iotrace_L4_nofreeze` | 4 | 7.45 GiB | off | completes, 5101 operations, gate written |

```
L=12   party2  IO|seq=2|op=write|want=14155776|got=2703424|errno=0
       party3  IO|seq=2|op=read |want=14155776|got=2703424|errno=0
L=4    party2  IO|seq=20|op=write|want=18874368|got=14340777|errno=0
       party3  IO|seq=20|op=read |want=18874368|got=11525008|errno=0
```

- Same `want` on both sides, partial `got`, `errno = 0`: a signal after some bytes moved
  (`MSG_WAITALL` does not prevent it). The signal is the freeze poller; more key material, more
  captures, earlier failure.
- The first no-freeze control passed `SNNI_PM_FREEZE=0` and failed: `poll_peak.sh` tests the flag
  with `-n` (presence). The sbatch now removes the variable via a named diagnostic switch.

Second defect: the completing runs wrote 98,304 zeros. The input fill `(v * (1<<scale)) / 1000000`
with `v` in [-1000, 1000] gives inputs in [-4, 4] at scale 12 (1.0 = 4096), numerically zero (same
defect as SHARK's first gate; SIGMA-GPU's patch documents it).

Fixes in the measured program: the socket layer transfers until complete (no protocol byte changed);
the fill divides by `1000` and announces `GATE_NONZERO` / `GATE_DEGENERATE`.
