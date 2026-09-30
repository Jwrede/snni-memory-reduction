# b1_dot_encode_chunk: first measurement

Measured 2026-07-30 on the placeholder-operator benchmark (jobs 45518635-37, derives 45518638-40,
`express`, policy `2026-07-29.1`). The published row is the re-measurement with the faithful
operators (`steps.csv`). Image `bumblebee-b1_dot_encode_chunk.sif`, binary md5
`5a772143c5256e276cc8bbf9ed8c00e6` (parent `6678529786c99a288c38b4ac188ee013`). `levers.env` is the
declaration before the runs.

| | b0_default | b1_dot_encode_chunk |
|---|---:|---:|
| peak_host | 15,446,988 kB | 8,613,808 kB |
| W_host | 38.01 | 21.17 |
| `seal::util::MemoryPoolHeadMT::get()` (grouped) | 8639.0 MB | 1649.0 MB |
| `yacl::Buffer::Buffer(long)` (grouped) | 6365.1 MB | 6472.6 MB |
| named_frac_host | 0.9671 | 0.9565 |
| spread_host_pct | 0.091% | 0.209% |

- Peak -6,833,180 kB (-44.2%); declared object -6990 MB.
- Wall 306 s median against 310 s; `runtime_delta_pct` +2.7% against a 6.46% spread: free. Possible
  mechanism: the peer's ciphertext receive no longer overlaps the whole-side encode.

## Gate

The declaration predicted a bit-identical result; not observable (Cheetah truncation is one-bit
approximate, shares come from OS entropy in thread-dependent order).

| comparison | worst absolute difference | ULP at `2^-16` |
|---|---:|---:|
| b0's replicates against each other | 5.07e-3 | 332 |
| b1's replicates against each other | 4.87e-3 | 319 |
| b1 against b0, all nine pairs | 5.25e-3 | 344 |

## Transcript

| | rank 0 sent, median of 3 | spread within the step |
|---|---:|---:|
| b0_default | 6,174,980,450 B | 8.50e-7 |
| b1_dot_encode_chunk | 6,174,975,221 B | 1.46e-6 |

Difference 5229 B (8.47e-7), below b1's own spread (SEAL's compressed ciphertext serialisation).

## Not established

- Arithmetic exactness of the slicing (argued from source: each `out[i,k]` accumulates over `j` in
  ascending order; slicing reorders only independent accumulators).
- Layers 2 to 12 (the gate reads layer 1; the lever sits in the matmul routine all linear operators
  call).
