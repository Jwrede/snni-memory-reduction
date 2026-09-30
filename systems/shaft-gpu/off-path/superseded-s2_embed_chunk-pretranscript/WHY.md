# s2_embed_chunk before the transcript

Jobs 45635528-30, 3 replicates, declarations conformant, gate failed.

| | s1_limb_loop | s2_embed_chunk | |
|---|---:|---:|---|
| VRAM peak, kB | 2,406,400 | 1,546,240 | -860,160, -35.7% |
| host peak, kB | 3,412,564 | 3,262,892 | -149,672, -4.4% |
| gate, first two logits | `3.1565629440e+09` `-7.9492614400e+08` | `1.9500732422e-01` `1.0627746582e-01` | worst relative difference 1.000e+00 against 1.000e-06 |

- Chunking the vocabulary-wide one-hot matmul into 16 parts draws a Beaver triple and PRZS masks
  per chunk; truncation error depends on the masks, so the output changes
  (`systems/shaft-cpu/GATE-TOLERANCE.md`).
- The baseline output on this machine, `3.157e+09`, is a fixed-point wraparound. The chunked
  variant gives `1.950e-01 / 1.063e-01`, within 0.6% of its MIG-slice value
  (`1.9624e-01 / 1.0294e-01`). Seed, device and work partitioning each move the 12-layer result.

## 1-layer gate (jobs 45639682/83, `SHAFT_N_LAYERS=1`)

| | logit 0 | logit 1 |
|---|---|---|
| N=1, baseline | 1.4225769043e-01 | 2.2676086426e-01 |
| N=1, this lever | 1.4314270020e-01 | 2.2621154785e-01 |

- Worst difference 8.85e-4 absolute, 6.2e-3 relative.
- Mechanism estimate: 2^-16 x ~10 truncations per layer x layer gain ~10 = ~1e-3 absolute.
- Comparison: BumbleBee 6.0e-6 relative (1-layer gate holds), SHAFT-CPU 2.06 (does not).
- Not adopted: two unchanged 12-layer runs here are byte-identical, so the 12-layer gate works.
  The lever was admitted through phase B instead.
