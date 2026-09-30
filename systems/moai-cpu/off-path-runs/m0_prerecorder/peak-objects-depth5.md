# m0_default at recorder depth 5

`peak-objects.md` beside this file (median replicate, depth 1) is one row:

```
409831.8 MB  99.3%  heap:test+0x115e0d(near seal::util::MemoryPoolHeadMT::get())
```

Run 3 with `SNNI_PM_DEPTH=5`: job 46099818, `bigsmp` r05n01, image digest `6b14ce49...`,
`OMP_NUM_THREADS=16`, sampled at 403,287,264 kB (99.995% of that run's 403,306,200 kB peak).
Attributed 412,928.5 MB.

| object | size | frac |
|---|---:|---:|
| `seal::Ciphertext::operator=(seal::Ciphertext const&)` <- `MemoryPoolHeadMT::get()` | 228,647.9 MB | 55.4% |
| `seal::Ciphertext::resize(SEALContext const&, array<unsigned long,4>, ...)` | 153,244.5 MB | 37.1% |
| `seal::Evaluator::mod_switch_to_next(...)` | 10,244.1 MB | 2.5% |
| `seal::Evaluator::relinearize_internal(...)` | 7,024.4 MB | 1.7% |
| `seal::Evaluator::apply_galois_inplace(...)` | 4,960.0 MB | 1.2% |
| `seal::util::CreateNTTTables(...)` | 2,924.4 MB | 0.7% |
| `libc.so.6+0x9cb84` <- `MemoryPoolHeadMT::get()` | 751.9 MB | 0.2% |
| `seal::Evaluator::rotate_internal(...)` | 671.6 MB | 0.2% |
| `libgomp.so.1.0.0+0x2279f` <- `MemoryPoolHeadMT::get()` | 618.2 MB | 0.1% |
| `anon:[heap]` | 379.9 MB | 0.1% |

- Basis of `m1_boot_out_move` (rank 1: ciphertext copy-assignment).
- Replicates differ only in recorder depth; peaks within 0.153%:

```
run1 / run2  (depth 1)   402,690,024 and 402,965,064 kB
run3         (depth 5)   403,306,200 kB
```

- `make_steps.py` publishes the median's own table; this is a second table from a third execution.
- Residual: subtraction 590.8 MB, direct location 536.4 MB (54.4 MB gap, 0.014%, flagged by the
  derive). 2.7 GB `PROT_NONE` reservation not scanned (measured empty; re-check with
  `SNNI_PM_FULLSCAN=all`).
