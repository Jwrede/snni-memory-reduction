# s1_arena_cap: not a step

Moved off the line 2026-08-14. Runs, logs and declaration unchanged.

| | |
|---|---|
| lever | `MALLOC_ARENA_MAX=2`, 3 replicates, gate byte-identical |
| peak | -227.4 MB |
| `anon:` total | -245.8 MB (`mmap-7` to `mmap-11`, `mmap-rest`: glibc arenas) |
| `heap:` total | -6.7 MB; no named object moves |

Chosen when 94.1% of the peak was unnamed (the image had no `binutils`; every `addr2line` failed
silently). After `symbolise.py` gained its ELF-symbol-table fallback, the same runs show a ranked
object at 72.2% that this lever does not touch:

```
NON-CONFORMANT OBJECT s1_arena_cap: attacked 'anon:mmap arenas (glibc per-thread arenas capped
to 2)' but the largest object at the previous peak was 'heap:pybind11::bytes::bytes...'
```

`MALLOC_ARENA_MAX`-type measurements across systems: SHAFT-CPU `p_alloc_probe` (-2.2% peak, +35%
wall), BumbleBee `x4_seal_pool_new` (+12.7%).
