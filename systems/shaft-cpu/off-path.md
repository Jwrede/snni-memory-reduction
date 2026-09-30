# SHAFT-CPU: off-path

Measured changes outside the line `s0`-`s8`, and where the line ends. Run directories:
`off-path/<run>/` (step-shaped: `levers.env`, `logs/`, `peak-objects.md`) and
`off-path-runs/<run>/` (with `RESULT.md`).

## Status: the line is closed at `s8`

| | |
|---|---|
| peak | `s0_default` 3,624,736 kB, `s8_encrypt_chunk` 851,224 kB, -76.5%, 4.258x |
| `W` | 16.03 -> 3.30 |
| rows | baseline, 5 `free`, 2 `marginal` (`s5`, `s6`), 1 `paid` (`s8`: +4.0% wall at 1.9% spread) |
| measured | 2026-09-29, `freeze_mode=maxparty`, recorder c2-32b38f744, input tokens drawn from the cased vocabulary (0..28995) |
| order | spill moved into `s4_onnx_and_spill`, so no free step follows a paid one (rule 6); argument in `steps/s4_onnx_and_spill/levers.env` |

### End of the line

| at `s8`'s peak | MB | frac | eligible |
|---|---:|---:|---|
| `anon:residual` | 623.5 | 0.715 | no: `anon:` is located, not identified |
| `heap:torch:torch.int64 held_by=dict` | 100.8 | 0.116 | target term (table share, Beaver scratch); only nameable heap object above 7 MB |
| `file:libtorch_cpu.so` and other `file:` | ~123 | 0.13 | library overhead, excluded from `W` |

`named_frac_host` 0.2847, `object_vs_unnamed` 0.16.

| remaining option | measured | disposition |
|---|---|---|
| allocator returns its pages | `off-path-runs/p_alloc_s8`: peak -12.48%, replicate spread 8.59% -> 0.31%; `anon:` -194.0 MB against a -112.6 MB peak fall, no named object reduced | allocator setting; acts on `anon:` only |
| reduce the int64 working set | -- | target term |
| low-degree polynomial GELU | not measured | accuracy trade |

Peak instant, measured on the `s8` runs before the 2026-09-29 re-measurement
(`p_graphop_s8`, `graphop_markers_cpu`, per-node `VmRSS` and `VmHWM`; see
`off-path-runs/s9_encrypt_chunk_inplace/RESULT.md`):

- 91.8% of the peak is reached before the first graph node runs.
- The high-water mark changes in 3 of 15 subgraphs; only the GELU raises it, by 8.2%.
- 29 high-water rises over the run, 616,716 kB in total, largest 11%, last in layer 7.
- `s9_encrypt_chunk_inplace`: at its own site +126,076 -> +5,676 kB, peak -0.7%.

## Candidates refused for `s4`

| run | change | result | reason |
|---|---|---|---|
| `off-path/x_onnx_at_beaver_peak` | ONNX export fix alone | -2.16%, spread 9.76% | lowers the non-binding one of two equal humps |
| `off-path/x_spill_at_convert_peak` | spill alone | +380 kB, spread 0.386% | lowers the other hump |
| `off-path/x_encrypt_on_convert_bimodal` | encrypt-on-convert | -4.89%, spread 9.867% | same bimodality; the attacked row held a different object at this peak |

The 9.76% and 9.867% spreads are replicates landing on one hump or the other.

## The constraint that shapes the lever space on this system

| | |
|---|---|
| mechanism | CrypTen fixed-point truncation depends on the random masks |
| measured | SHAFT-GPU, unseeded, 5 runs of one image: first logit spans 3.8x, one run -3.47e+09 (wraparound); `GATE-TOLERANCE.md` |
| consequence | a lever that changes random draws fails the 1e-6 gate; it can only be a phase B step (same exchanged bytes, plaintext equivalence test, README.md rule 7) |
| phase B steps | `s5_per_layer`, `s6_stream_load`, `s7_embed_chunk`, `s8_encrypt_chunk`; tests in `impl/equiv_*.py` |

Off-path on this system regardless of size:

| mechanism | reason |
|---|---|
| polynomial GELU in place of the 8-term Fourier series | changes the computed function |
| skipping the discarded triple on the non-source party | protocol redesign (below) |
| fewer threads | 6.2 s of 224.5 s in the earlier campaign; per-thread PRNG; thread count pinned |

## Not attempted: the discarded triple on the non-source party

- `TrustedFirstParty.generate_additive_triple` draws `a`, `b`, `c` from the `local` generator on
  both parties; `ArithmeticSharedTensor(..., src=0)` uses them only on rank 0 (`arithmetic.py:103`).
- Party 1 materialises a (28996, 768) and a (1, 128, 28996) int64 tensor, computes their matmul and
  discards all three.
- Not drawing them shifts party 1's `local` generator, so every later share and the logits change.
  The fix (source party draws, the other derives its share from a shared seed) is a protocol change.

## Disposition: the encrypted parameter set, skipped at `s2`

| | |
|---|---|
| declared by | `s2_triple_share_free`, `object_skipped: encrypted parameter set` |
| object | 862,347,264 B over the graph's 201 `Parameter` modules |
| only fix | the spill (create each share at first use) |
| cost when `s2` was chosen | paid: +1.4% wall against 0.441% spread, n=3; memory -3.6% |
| disposition | rule 6 defers it behind the free fixes; it enters the line in `s4_onnx_and_spill` |

## Skipped by `s8_encrypt_chunk`: the embedding table share

| | |
|---|---|
| object | `heap:torch:torch.int64 held_by=dict`, 178.2 MB at `s7`'s peak (run3's frozen table) |
| content | word-embedding table as int64 shares, 28,996 x 768 x 8 B |
| disposition | target term (table share), required during the lookup; `s8` attacks the next row, the float32 plaintext of the same table |

## Diagnostic: a second input (`off-path-runs/p_input2`)

| step | change in peak | spread |
|---|---:|---:|
| `s0` | -0.44% | 0.49% |
| `s8` | +1.8% | 15.9% (same session) |

Input token seed shifted, compared with a same-session rerun of the published input. Both changes
are inside the spread.

## Accuracy / security / portability trades

None taken. The polynomial GELU (4th degree, ~192 MB per call in the earlier campaign) is not
measured.

---

# Earlier stages of this line

Runs measured before the published ordering or before the 2026-09-29 re-measurement. Step names
are those of the ordering that measured them. Peaks in kB, median of 3 replicates.

## Measured changes

| run | base | change | peak | spread | wall | verdict |
|---|---|---|---|---:|---:|---|
| `off-path/s3_graph_clear_values` | `s2` | free graph values after use | 2,950,548 -> 2,960,800 | 0.31% | | peak unchanged; RSS at `after_inference` -386 MB |
| `off-path/s4_graph_clear_values` | `s3` | same | 2,670,192 -> 2,680,744 | 2.59% | | same |
| `off-path/p_alloc_probe` | `s3` | `MALLOC_ARENA_MAX=1`, `MALLOC_TRIM_THRESHOLD_=0`, `MALLOC_MMAP_THRESHOLD_=131072` | 2,670,192 -> 2,611,260 (-2.2%) | | 102 -> 138 s (+35%) | allocator retention at most 2.2% of that peak |
| `off-path/s4_trace_nograd_outoforder` | `s3` | trace under `no_grad` | -6.3% | | | `object_conformant=no`: attacked `libc10.so+0x4a675`, largest was `heap:python3.10+0x13cc27` (867 MB ONNX buffer) |
| layer_stream (run not retained) | `s1` | stream layers | -12.8% | | free | decomposition at 0.741 of the peak; the 201 shares still coexist |
| `off-path/s2_beaver_reveal_inplace` | `s1` | remove two redundant (28996, 768) int64 share copies (`arithmetic.py:359,380`, `distributed_communicator.py:195`) | 3,059,428 -> 3,024,540 (-1.15%) | 1.171% | | inside spread; `VmHWM` final at `after_triple`, before the reveal |
| `off-path/s4_plaintext_free_early` | `s3` | release the plaintext model before `from_onnx` | 2,693,096 -> 2,706,244 (+0.49%) | 1.307% | | a second hump of equal height |
| `off-path/s4_param_stream` | `s3` | spill all 201 encrypted parameter shares (866,494,480 B) to disk, read at their node | 2,693,096 -> 2,673,336 (-0.74%) | 0.812% | +1.9% | inside spread; freed 2.36 MB chunks stay in the glibc arena (`anon:[heap]` 383.6 -> 555.3 MB) |
| `off-path/s4_onnx_file_export` | `s3` | ONNX export to a file instead of `io.BytesIO` | 2,693,096 -> 2,619,636 (-2.7%) | 1.804% | +2.2% | `object_conformant=no`: largest object at `s3`'s peak was `heap:libc10.so+0x4a675` |
| `off-path/s4_param_release` | `s3` | spill plus global `MALLOC_MMAP_THRESHOLD_=131072` | 2,693,096 -> 2,621,100 (-2.67%) | | +41.1% (spread 3.571%), paid | global threshold routes every Beaver temporary through `mmap`; scoped variant `s4_param_mmap_release` |
| `off-path/s8_plaintext_free_early` | `s7_graph_clear_values` | release the caller's plaintext model before the parse | 2,061,808 -> 2,060,524 (-0.06%) | 1.364% | -0.8% | object halves (865.8 -> 433.5 MB), peak unchanged: second copy at `nn/onnx_converter.py:183`, freed pages reused |
| `off-path/s8_encrypt_on_convert` | `s7_graph_clear_values` | also encrypt each initializer at creation | -0.47% | 1.41% | -1.3% | inside spread |
| `off-path/s8_convert_release` | `s7_graph_clear_values` | also low `M_MMAP_THRESHOLD` across `from_onnx`, then trim | -0.48% | 0.801% | +1.6%, marginal | inside spread; `malloc_trim` returned 315,116 / 352,908 / 353,132 kB |
| `off-path-runs/p_freezeall_s5` | `s5`, runs of 2026-08-18 | `SNNI_PM_FREEZE_ALL_POLLERS=1` (every poller stops both parties) | 1,711,156 -> 1,590,008 | 0.594% -> 6.969% | stopped 6 s -> 62-77 s | aligned tables for both parties in 3/3 runs; not published |

## End of the 2026-07-30 ordering: the peak is inside `torch.onnx.export`

| | |
|---|---|
| line | `s7_graph_clear_values`, 3,594,240 -> 2,061,808 kB, 1.74x |
| probe | `impl/levers/probe_export_trace_cpu.py`, job 45511828, trace file not retained |
| result | `VmHWM` reaches 99.5% of the run's peak (2,051,888 kB) inside one export call; ~1.24 GB transient, ~850 MB returned |
| held during export | parameters (433 MB), the traced graph's copies, the protobuf (`export_params=True`), ~1.3 GB |

```
EXPORT|export1_before            RSS 709588   hwm 1122440
EXPORT|export1_inflight_max_rss             1946004
EXPORT|export1_after             RSS 1099296  hwm 2041472
```

No free fix:

- `export_params=False` exports a graph without weights.
- torch 2.0.1 `torch.onnx.export` has no external-data option.
- `crypten.nn.from_pytorch` is defined as ONNX export plus `from_onnx`
  (`nn/onnx_converter.py:52-68`).

| object at `s7`'s peak | size | disposition |
|---|---:|---|
| `heap:_multiarray_umath...+0x13ab14` | 865.8 MB | removed by four attempts, peak unchanged: not at the peak instant |
| `heap:libc10.so+0x4a675` | 535.3 MB | encrypted parameter set and Beaver operands |
| `anon:[heap]` | 269.9 MB | allocator working set, never a target |
| `file:libtorch_cpu.so` and other `file:` | 133 MB | library overhead |

## Index of `off-path-runs/`

| run | content |
|---|---|
| `p_alloc_s8` | allocator pinned on `s8`'s lever set, n=3 |
| `p_freezeall_s5` | `s5` with every poller freezing both parties, n=3 |
| `p_input2` | second input at `s0` and `s8` |
| `per_layer_arm` | per-layer steps of the earlier campaign, ported and gated |
| `per_layer_arm_refutations` | four per-layer variants, n=3 each |
| `s5_embed_chunk_wholemodel` | embedding chunking applied on `s4`, whole model loaded |
| `s5_graph_clear_values` | graph value release at a second operating point |
| `s6_conversion_phase` | three changes to the conversion phase |
| `s9_encrypt_chunk_inplace` | in-place chunk encryption on `s8` |
| `vit_transfer`, `vit_v0_default`, `vit_v1_endpoint` | ViT-Base/16 runs (`systems/shaft-cpu-vit`) |
