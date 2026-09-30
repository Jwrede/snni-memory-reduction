# s3_modswitch_lastlimb: segfault

Job 46120481, `gpuh200`, `s2_boot_input_release`'s tree plus `patch_modswitch_lastlimb.py` (beside this
file).

Every mod-switch copies its input first:

```c++
auto encrypted_copy = make_cuda_auto_ptr<uint64_t>(encrypted_size * coeff_mod_size * poly_degree);
cudaMemcpyAsync(encrypted_copy.get(), encrypted.data(), ...);
destination.resize(context, next_index_id, encrypted_size, stream);
rns_tool.divide_and_round_q_last_ntt(encrypted_copy.get(), ..., destination.data(), ...);
```

| | |
|---|---|
| row | 33,843.8 MB, 30.5% of the attributed peak (rank 2) |
| reading | `divide_and_round_q_last_ntt` starts with `nwt_2d_radix8_backward_inplace(ci_in, rns_tables, 1, base_q_size - 1, stream)`, taken to mean only the last limb is written |
| change | save the last limb, pass `encrypted.data()` directly, restore the limb |

```
signal 11 (segmentation fault, core dumped), wall 1780 s
FAILED: level trace has 7 of the 12 transitions a complete run emits
FAILED: 11 of 16 phase markers
FAILED: no GPUPEAK line
```

- Lever active (`LEVER|modswitch_lastlimb|limbs=15`, later `limbs=3`); CUDA errors in `stderr.log` are
  `driver shutting down` (teardown).
- Eliminated: aliasing of `destination` and `encrypted` (both call sites, `evaluate.cu:1551` and
  `:1582`, declare a fresh local `PhantomCiphertext destination;`).
- Remaining hypothesis: the function writes more than the last limb. Not pursued; the copy is required.
- Next at the time: `s3_gelu_res_move`.
