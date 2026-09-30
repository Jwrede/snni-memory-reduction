# b2_result_ct_pack_free: first measurement

Measured 2026-07-30 on the placeholder-operator benchmark (jobs 45519462-64, derives 45519465-67,
`express`, policy `2026-07-29.1`). The published row is the faithful re-measurement (`steps.csv`).

| | b1_dot_encode_chunk | b2_result_ct_pack_free |
|---|---:|---:|
| peak_host | 8,613,808 kB | 7,520,936 kB |
| W_host | 21.17 | 18.48 |
| `seal::util::MemoryPoolHeadMT::get()` (grouped) | 1649.0 MB | 528.1 MB |
| `yacl::Buffer::Buffer(long)` (grouped) | 6472.6 MB | 6365.1 MB |
| named_frac_host | 0.9565 | 0.9409 |
| spread_host_pct | 0.209% | 0.134% |

- Peak -1,092,872 kB (-12.7%); declared object -1121 MB.
- `runtime_delta_pct` -3.6% against a 3.945% spread: free.
- `object_conformant: yes-skipped` (`yacl::Buffer`, 6472.6 MB, declared `# object_skipped:`;
  `b3` later showed that row merges two objects, see `../b3_weight_defer/RESULT.md`).

## Falsification criterion

The response is `CeilDiv(out_n, 64)` ciphertexts either way; a transcript change beyond SEAL's
compression noise would void the row.

| | rank 0 sent, median of 3 | spread within the step |
|---|---:|---:|
| b1_dot_encode_chunk | 6,174,975,221 B | 1.46e-6 |
| b2_result_ct_pack_free | 6,174,974,544 B | 2.12e-6 |

677 B apart (1.10e-7). Gate: worst 5.249e-3 against the baseline, 5.066e-3 within b2.

## Peak location (median run, run3)

| transition | b0 | b1 | b2 |
|---|---:|---:|---:|
| `layer begin` -> `mha begin` | +3.09 GB | +3.08 GB | +3.09 GB |
| `mha begin` -> `after_qkv_linear` | +6.29 GB | +1.25 GB | +0.45 GB |
| `after_qk_matmul` -> `after_softmax` | +2.60 GB | +2.62 GB | +2.60 GB |
| `ffn begin` -> `after_up_linear` | +2.08 GB | +0.31 GB | +0.01 GB |

The unchanged transitions fault in the Ferret OT buffers (together 5.69 GB = 16 x 2 x 160 MiB).
At b2's peak: ~5.7 GB Ferret OT, ~0.8 GB model input, ~0.5 GB SEAL pool (one 256 MiB encode slice
dominant, a budget introduced by b1).
