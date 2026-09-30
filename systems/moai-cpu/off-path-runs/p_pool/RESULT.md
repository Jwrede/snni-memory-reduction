# p_pool: SEAL memory manager in the baseline

Jobs 45843893 (wrong root), 45843941 (source introspection in the measured image), 2026-08-06; follow-up
job 45844903.

```
heap:test+0x115e0d(near seal::util::MemoryPoolHeadMT::get())   409831.8 MB   0.993 of the peak
```

- The measured program never uses SEAL's `MemoryManager` (`MemoryManager`, `MMProf`, `SwitchProfile`,
  `MemoryPoolHandle` appear under `/root/moai` only in SEAL's dotnet bindings): it runs the global
  profile, which retains every block. The peak is the pool's high-water.
- Consequence: object releases return pages to the pool; the allocator policy has to change first.
- The build had one marker call site at the time; a run took 22 h 45.
- Earlier campaign's fix (its snapshot): "EXP12 fix: use MemoryPoolThresholdMT instead of
  MemoryPoolHandle::New(true)." `CANONICAL_RESULTS.md`: 72.3 GiB on two layers with `MMProfThreshold`
  among three mechanisms.

| SEAL profile (this fork) | behaviour |
|---|---|
| `MMProfGlobal` | the default, one global `MemoryPoolMT` that retains everything |
| `MMProfNew` | a brand-new pool per `GetPool()` call, so nothing is retained across calls |
| `MMProfFixed` | a caller-supplied pool, still a `MemoryPoolMT`, so still retaining |
| `MMProfThreadLocal` | one retaining pool per thread |

- `MemoryPoolThresholdMT` does not exist in this fork; the earlier campaign wrote it.
- Open at the time: retained-dead against simultaneously-live share of the 384.29 GiB (measured later:
  `../p_prm0/`).
- Probe defect: the first version searched `/opt /usr/src /src`; `ROOT` stayed empty and grep searched
  the bind-mounted campaign directory, finding `src_snapshot/` (the earlier campaign's instrumented
  tree) with `MemoryPoolThresholdMT`, `MMProfFixed`, phase-split keys. The probe now refuses if the tree
  is missing.
