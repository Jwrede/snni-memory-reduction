# s3_gelu_res_move: buffer changed owner

Job 46120517, one replicate, image layered on s2's tree. Patch: `impl/patch_gelu_res_move.py`.

| | |
|---|---|
| change | `gelu_other.cuh:179`, `res = x_n[i]` at `i == 1` -> `res = std::move(x_n[i])` |
| reason | row 6442.5 MB at s2's peak; `x_n[1]` dead at the assignment; `cuda_auto_ptr::operator=(const cuda_auto_ptr &)` does `cudaMallocAsync` plus `cudaMemcpyAsync` (deep copy); `PhantomCiphertext` has a defaulted move assignment |
| marker | `LEVER|gelu_res_move|armed` |

```
                        peak_alloc_kb    peak_reserved_kb   wall s   gate
s2_boot_input_release   108,321,427      123,928,576        3342     01469da3...
s3_gelu_res_move        108,321,427      143,491,072        3349     01469da3...
```

Published `peak_vram` (reserved) +19,562,496 kB (+15.8%); allocated identical. Off-path.

Attribution channel (46120518):

```
                     gelu_other.cuh:179   mod_switch_scale_to_next   attributed total
s2_boot_input_release      6442.5 MB              33843.8 MB            110921.1 MB
s3_gelu_res_move              0.0 MB              40286.3 MB            110921.1 MB
                                                   +6442.5 MB
```

- `res` now holds the buffer allocated by `mod_switch_scale_to_next` (via `rescale_to_next_inplace`);
  one buffer instead of two, attributed elsewhere. A per-site table can move 6.4 GB between rows with
  no change in the total.
- `peak_reserved_kb` moved 15.8% with an identical allocated high-water: reserved depends on
  fragmentation and pool growth order. `s2` stands (allocated 119,052,371 -> 108,321,427 kB alongside
  reserved -12,550,144 kB).
