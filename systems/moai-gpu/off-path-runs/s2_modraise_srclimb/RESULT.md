# s2_modraise_srclimb: refuted; the row was never the copy

Measured 2026-08-07: jobs 45906784 (clean), 45906785 (`_attrib`), 45906786 (`_pmrec`), 45906787
(derive). Earlier ordering.

| | |
|---|---|
| change | in `Bootstrapper::modraise_inplace`, `PhantomCiphertext cipher_copy = cipher` replaced by a buffer holding only the limbs the kernels read (`4 x src_limb` in the built tree) |
| prediction | site 28,185.7 MB -> ~800 MB (~21% of a 129.19 GiB peak) |

| | s1 | s2 |
|---|---:|---:|
| VRAM `peak_alloc_kb` | 119,052,371 | 119,052,371 |
| VRAM `peak_reserved_kb` | 135,462,912 | 137,789,440 |
| host peak, kB | 2,106,392 | 2,104,356 |
| wall, s | 3,374 | 3,345 |
| the attacked row | 28,185.7 MB | 28,185.7 MB |

```
s1   26880.0 MiB   count=768    (published as 28,185.7 MB)
s2   26880.0 MiB   count=768
```

- 768 allocations of 35 MiB = `2 polys x 35 limbs x 65536 coeffs x 8 B` each: the ciphertext's own
  resize to full level in the mod-raise. The copy was removed but was not this row (that would show
  `count=1536`).
- A symbolised row is keyed on the enclosing function (the allocation is in `cuda_runtime.h:745`), so
  it aggregates every allocation in that function. `size / count` identifies the object without a run.
- Also: `make_steps.py` reported `attacked=vram but required=host` (host `W` larger at s1's peak).

## Recorder: wrapped per-site balances

```
18446744073708.5M   count 18446744073709551516   check<cudaError>     (= -1.5 MB, count -100)
18446744073621.5M   count 18446744073709551604   check<cudaError>     (=  ...    count  -12)
18446744072598.1M   count 18446744073709551421   check<cudaError>     (=  ...    count -195)
    75357.7M        count 17697                  check<cudaError>     the real one
...
    55340232295308.2 MB   check<cudaError>                            the aggregate
```

Several call sites resolve to `cuda_wrapper.cuh:26:check<cudaError>`; some saw more frees than
allocations (`untracked_frees=19` on s1, `25` on s2); unsigned balances wrapped. `cudarec.c` was
corrected on 2026-08-09 (MOAI-GPU re-measured; `systems/sigma-gpu/MANIFEST.md`, Instrumentation).
