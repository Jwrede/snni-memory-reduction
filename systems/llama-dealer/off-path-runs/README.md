# LLAMA dealer: off-path runs

Same procedure and gates as the line; `make_steps.py` reads only `steps/`.

| run | content |
|---|---|
| `d3t_truncate_free` | truncate lever on `d2`: -259,624 kB against the activation lever's -343,868 kB |
| `d5a_ars_free` | SlothARS scratch freed on `d4` |
| `d3_alloc_trim`, `d7_tensor_pool`, `d7_wrap_chunk`, `d7_sumsq_free` | rejected steps (`../off-path.md`) |
| `p_arena1`, `p_mmap_thresh`, `p_jemalloc`, `p_floor4k`, `p_omp1`, `p_omp4`, `p_omp8` | floor mechanisms (`../off-path.md`) |
| `p_diag`, `p_fast15`, `p_freeze`, `p_sync`, `p_trace` | diagnostics |

## d5a_ars_free

| lever | object, grouped by function | measured drop |
|---|---:|---:|
| SlothARS freed (this run) | 76.9 MB | -74052 kB |
| SlothGelu freed (published d5) | 173.2 MB | -169136 kB |

Both gate clean; each drop is close to its object's grouped size.
