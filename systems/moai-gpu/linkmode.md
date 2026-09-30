# MOAI-GPU: static against dynamic CUDA runtime

Result: indistinguishable on the device peak; the live high-water is byte-identical in all six runs.
The dynamic link is a declared property of the measured system (baseline and every step). Measured
2026-07-31, jobs 45520648 (build), 45520649 (six runs); decision rule fixed in
`impl/diag_compare_linkmode.py` before the runs.

| | |
|---|---|
| problem | upstream links `libcudart_static.a`: `cudaMallocAsync` defined in the binary (`T cudaMallocAsync`), `readelf -d` lists neither libcudart nor libcuda, no undefined `cu*` symbols, no cupti; `LD_PRELOAD` cannot interpose |
| change | `CMAKE_CUDA_RUNTIME_LIBRARY=Shared` |

| | needed cudart | `cudaMallocAsync` defined | undefined, i.e. interposable |
|---|---|---|---|
| `build-static` | 0 | 1 | 0 |
| `build-shared` | 1 | 0 | 1 |

Three replicates each (earlier campaign: 1.5 to 2.5 GiB fragmentation band across runs of one binary):

| | static | dynamic |
|---|---|---|
| reserved high-water, kB | 134184960 / 135888896 / 134807552 | 137658368 / 134184960 / 136380416 |
| polled, MiB | 131715 / 133379 / 132323 | 135107 / 131715 / 133859 |
| live high-water, kB | 119052371 x3 | 119052371 x3 |
| wall, s | 3343 / 3345 / 3339 | 3372 / 3363 / 3367 |

```
reserved   spread static 1.26%   spread dynamic 2.55%   between medians 1.17%   INDISTINGUISHABLE
polled     spread static 1.26%   spread dynamic 2.53%   between medians 1.16%   INDISTINGUISHABLE
```

- The 1.17% median difference is also below the static spread alone.
- Wall: dynamic +0.72% at the median, rank-separated (PLT indirection); cancels in `cost_type`.
- Reserved and polled differ by a constant ~0.66 GiB (CUDA context, seen by `nvidia-smi`, not the pool).

## Consequences

- Device recorder as the CUDA analogue of `pmrec.so`: `LD_PRELOAD` on `cudaMallocAsync`,
  `cudaFreeAsync`, `cudaMalloc`, `cudaFree`; records pointer, size, call site; snapshot at each new
  maximum; emits `objects_vram.jsonl`.
- Binary rebuilt with `-g` (it had no `debug_info` sections).
- Live and reserved are kept apart: 113.5 GiB live against 128.0 GiB reserved at one instant.
