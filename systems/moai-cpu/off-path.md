# MOAI-CPU: off-path

## Status: the line is closed at `m26`, above its target

27 rows, one replicate each (`steps.csv`). No trades. Declared scope: two of twelve layers
(MANIFEST.md, RUN_META, `WORKLOAD|` marker), applied to every step.

| step | peak (kB) | `W` | change | cost | conformant |
|---|---:|---:|---:|---|---|
| `m0_default` | 402,542,092 | 15.93 | | baseline | |
| `m1_pool_threshold` | 298,351,628 | 11.81 | -25.88% | paid | yes |
| `m2_minimal_att_keys` | 276,590,616 | 10.95 | -7.29% | free | yes |
| `m3_x_copy_release` | 251,374,288 | 9.95 | -9.12% | free | yes |
| `m4_enc_ecd_x_release` | 229,447,752 | 9.08 | -8.72% | free | yes |
| `m5_att_keys_release_last_layer` | 223,988,072 | 8.86 | -2.38% | free | yes |
| `m6_boot_layer_release` | 208,303,932 | 8.24 | -7.00% | free | yes |
| `m7_phase_split` | 153,463,664 | 6.07 | -26.33% | paid | yes |
| `m8_chunked_bootstrap` | 127,622,812 | 5.05 | -16.84% | paid | yes |
| `m9_ct_capacity_shrink` | 128,774,688 | 5.10 | +0.90% | free | yes-plateau |
| `m10_dead_boot_keys_release` | 113,601,652 | 4.50 | -11.78% | free | yes |
| `m11_tiled_ffn` | 101,976,260 | 4.04 | -10.23% | paid | yes |
| `m12_rtn_release_ln1` | 101,323,468 | 4.01 | -0.64% | free | yes-plateau |
| `m13_rtn2_release_ln2` | 97,898,620 | 3.87 | -3.38% | free | yes |
| `m14_enc_X_v_shrink` | 95,174,904 | 3.77 | -2.78% | free | yes |
| `m15_layernorm_release` | 89,142,276 | 3.53 | -6.34% | free | yes |
| `m16_park_att_keys` | 88,922,720 | 3.52 | -0.25% | paid | yes-plateau |
| `m17_park_x_copy` | 79,028,988 | 3.13 | -11.13% | paid | yes |
| `m18_copy_after_switch` | 72,799,100 | 2.88 | -7.88% | free | yes |
| `m19_att_keys_park_softmax` | 72,536,768 | 2.87 | -0.36% | paid | yes-plateau |
| `m20_park_boot_layer` | 69,808,580 | 2.76 | -3.76% | paid | yes |
| `m21_boot_input_spill` | 65,382,812 | 2.59 | -6.34% | paid | yes |
| `m22_x_copy_release_early` | 62,768,960 | 2.48 | -4.00% | free | yes |
| `m23_pool_threshold_256k` | 57,429,056 | 2.27 | -8.51% | paid | yes |
| `m24_x_stream_matmul` | 56,660,876 | 2.24 | -1.34% | paid | yes-plateau |
| `m25_x_copy_park_direct` | 55,104,816 | 2.18 | -2.75% | free | yes |
| `m26_att_keys_late_create` | 53,999,084 | 2.14 | -2.01% | free | yes |

7.455x (402.5 GB -> 55.3 GB decimal). Target 25,267,200 kB (25.9 GB, MANIFEST.md); not reached.

### End of the line

At `m26`'s peak (`steps/m26_att_keys_late_create/peak-objects.md`):

| | MB | frac |
|---|---:|---:|
| bootstrap key stage loaded by `stage_load_boot_phase` (`KSwitchKeys::load`) | 19,820.2 | 35.8% |
| bootstrapping-library scratch among the 15 largest rows (`Polynomial.cpp` `homomorphic_poly_evaluation`, `Bootstrapper.cpp` `bsgs_linear_transform`, `ModularReducer.cpp` `modular_reduction`) | 21,069.2 | 38.1% |
| `sealpool:retained` | 2,591.0 | 4.7% |
| largest object of the program's own data flow (`all_layer_test`, `test_full_scheme.hpp:1404`) | 1,613.7 | 2.9% |

- The excess above the target is library scratch and allocator retention (README.md rule 6).
- `m27`, bootstrap chunk 64 on `m26` (`off-path-runs/m27_boot_chunk_64/`): -3.7% and -7.5% against
  `m26`'s run. It slices a target term (the chunk in flight; the target fixes chunk 128 from `m8`):
  off-path. The peak moves into the layer-norm window.

## Free phase at the baseline (2026-08-27)

`p_prm2`: live decomposition at all 29 phase markers of one run. At every marker 97-98% of the live
set is one row whose allocating half is `seal::Ciphertext::operator=`; the program frame above it
changes as blocks are recycled:

```
layer 0, after_bootstrap1     178.7 GB   98.0%   test_full_scheme.hpp:676 <- Ciphertext::operator=
layer 1, after_bootstrap1     178.7 GB   98.0%   Bootstrapper.cpp:2130 <- Ciphertext::operator=
nearest the peak              232.8 GB   98.4%   gelu_others.hpp:143 <- Ciphertext::operator=
```

The high-water is built in layer 0 (layer 1 adds 0.1%):

```
phase                              RSS      live   retained by the pool
key generation                +121.8 GB  +115.8 GB      +6.1 GB
attention                      +42.7 GB    +1.1 GB     +41.6 GB
bootstrap 1                    +83.5 GB   +13.4 GB     +70.1 GB
layernorm 1                    +30.9 GB   +16.0 GB     +14.9 GB
FFN matmul                     +31.2 GB   +31.0 GB      +0.1 GB
GELU                            +6.7 GB   -25.2 GB     +31.9 GB
bootstrap 3 and layernorm 2    +32.7 GB   +12.3 GB     +20.2 GB

402.1 GB peak  =  159 GB key bank and context, resident before layer 0   40%
               +   72 GB the layer's own live working set                18%
               +  171 GB allocated, released, and never returned         42%
```

```
gelu_output          released in full, gate bit-identical, peak +0.26%      refuted
the x copy           released, gate bit-identical, peak -0.12%              refuted
object release generally  no MemoryManager in this program, global profile  refuted 2026-08-06
attention keys       m1 took the 17 of 30 that are never requested          taken, -5.31%
bootstrap keys       `KEYAUDIT|gal_steps_vector|requested=47` against
                     `gal_keys_boot|keys=47`: none is unused                nothing to take
holding one key set  `Bootstrapper` takes `gal_keys_boot` BY REFERENCE and
                     holds it for the whole run, and `gal_keys` is used in
                     every layer, so dropping either needs a reload         not free
```

On a pool that returns nothing to the OS, a release moves an object to the free list; the paid pool
repair (`m1`) goes first because it enables every release fix.

## The skipped object at m0: `gelu_output`

| | |
|---|---|
| row | `gelu_others.hpp:143:gelu_v2 <- seal::Ciphertext::operator=`, 234,877.6 MB, 57.0% of `m0`'s peak |
| object | `gelu_output`, 3,072 ciphertexts filled at `test_full_scheme.hpp:976`, last read at line 1019 (inlining puts GELU's line in the row) |
| measured fix (`m1_gelu_output_release`) | `LEVER|gelu_output_release|ciphertexts=3072`; gate bit-identical over 7,680 values; peak 402,401,204 -> 403,454,212 kB (+0.26%) |
| bound | GELU itself adds 1.67% to the high-water; releasing returns blocks to the free list |

```
after_selfoutput -> after_bootstrap1     +83,496,480 kB    20.8% of the peak
before_attention -> after_attention      +42,674,784 kB    10.6%
after_bootstrap2 -> after_intermediate   +31,160,828 kB     7.7%
after_bootstrap1 -> after_layernorm1     +30,878,328 kB     7.7%
after_bootstrap3 -> after_layernorm2     +16,472,084 kB     4.1%
after_final -> after_bootstrap3          +16,164,416 kB     4.0%
after_intermediate -> after_gelu          +6,724,288 kB     1.67%
```

## Skipped objects on the re-freeze line (2026-09-21)

Zen4, 16-core cpuset, shared node, chain-keyed recorder.

| row | disposition |
|---|---|
| `sealpool:retained` (62.7 GB at m1, 6.8 GB at m8) | retention aggregate; ranked, never attacked; its parameter is `m1` |
| `create_galois_keys(std::vector<int> ...)` at `test_full_scheme.hpp` (62.1 GB, 47 bootstrap rotations; 79.3 GB from m2 on, with the 14 attention rotations at the same site) | live through every bootstrap; paid phase split, taken as m7 after m2-m6 |
| at m10 (skipped by m11) | attention keys 17.2 GB (live for the next attention; park, paid, later), packed-input copy 16.9 GB (dead until the residual add; park, paid, later), bootstrap1 output `rtn` 16.9 GB (live); the layernorm input copy (16.9 GB) is the largest with a free fix |

### Skipped at m0

`m1_pool_threshold` skips both key banks. The pool repair goes first because the stock pool returns
nothing to the kernel (`gelu_output_release` +0.26%, `x_copy_release` -0.12%, `boot_out_move` +4.36%,
all inside the spread). `minimal_att_keys` (-5.2% measured on m0) becomes rank 1 with a free fix at
m1's peak and is taken as m2 (-20.7 GB).

Exclusive-node runs of m0-m6 from the first publication are not in the repository (peaks within
2.5 GiB of the re-freezes; m4 +5.8 GiB; collapsed (site, caller) recorder).

### Withdrawn 2026-09-21: m6_att_output_release

Chosen from a misread pair in m5's table (the bootstrap4 loop writing the next layer's input);
`att_output` is 1.6 GB at that peak; -0.21%, inside the spread. The phase-split image was built on
this run's image and carries the inert edit (declared by `steps/m6_phase_split` of that line).
`off-path-runs/m6_att_output_release/`.

## Skipped objects on the corrected-order line (2026-09-25)

| at | row | disposition |
|---|---|---|
| m10 (skipped by m11) | `sealpool:retained` 50.7 GB (rank 1 at m10's GELU-window peak) | the FFN intermediate (3072 ciphertexts from `ct_pt_matrix_mul_wo_pre_large`, `test_full_scheme.hpp:1388`), consumed by GELU and retained; tiling (m11) removes the allocation that feeds it. Rows below (attention keys 17.2, `boot_layer` 16.9, `rtn` 16.9 GB) live at that peak, paid fixes later |
| m14 (position 15) | layernorm1 input copy (`layernorm.hpp:208`, 16,914,534,400 B) and residual copy of the layer input (`test_full_scheme.hpp:1029`, 16,914,530,304 B) | both free (release after the last reader); larger taken as m15; the other as m22 |

## Phase-level spread on shared nodes (2026-09-25)

Pairs with identical code up to the phase:

```
pair (identical code up to the phase)     phase                 GiB           spread
m5r  / m6r   (boot_layer acts after bs2)  Layer-0 attention     171.7 / 170.7  -0.6 %
m5r  / m6r                                Layer-0 bootstrap1    176.4 / 172.8  -2.0 %
m12  / m13   (rtn2 acts after LN2)        Layer-0 bootstrap2     97.5 /  98.6  +1.1 %
m10r / m11r  (rtn acts after LN1)         Layer-0 layernorm1     90.9 /  90.9   0.0 %
m20j / m21s  (spill acts inside bs2)      Layer-0 attention      66.6 /  66.5  -0.2 %
```

The inert-step guard at n=1 uses 2.0 pp. A step whose global peak moves less is inert unless it
declares its phase and that phase fell by more than the spread (plateau step, README.md step 5).
Earlier estimate: 0.53 pp from eight phase markers of two `m1` runs under two recorder generations
(`m1`'s -5.31% about fifteen times that band).

## Instrument history

### The peak is a plateau (2026-08-27)

Against `m0`'s peak of 402,401,204 kB:

```
layer_1_begin      401,910,092   -0.122%
after_final        402,171,088   -0.057%
after_bootstrap3   402,171,088   -0.057%
after_layernorm2   402,171,088   -0.057%
after_bootstrap4   402,391,984   -0.002%
```

- The poller freezes at the highest RSS and each run froze in a different phase, so m0, m1 and
  `m1_gelu_output_release` reported three different rank-1 objects.
- Two samples 38 s apart inside one phase agree row for row (203,019.2 MB / 7,486 items against
  203,929.8 MB / 7,574 items).
- Since 2026-08-27 the poller takes one decomposition per phase marker (`SNNI_MARKER_SAMPLE=1`).

### Pool attribution (2026-08-25)

`pmrec` attributed a pool chunk to its first requester permanently:

- `gelu_output` (234,877.6 MB, 57.0%) did not appear in pmrec's table.
- Keys are live early (80.2 GB in 2,124 items of 37.75 MB, `off-path-runs/p_poolrec/`) and returned
  to the pool before the peak.
- 163.2 GB of the 402.3 GB peak held for objects that no longer exist.
- Producer: `MANIFEST.md`; measurement `off-path-runs/p_prm0/`; source reading
  `off-path-runs/p_pool/RESULT.md` (2026-08-06).

### Depth window

`m1`'s fixed-depth table put 91.5% into `Ciphertext::operator=` (58.0%) and `Ciphertext::resize`
(33.5%). Cause: the image (`off-path-runs/p_m1depth/`):

```
machine    image                      rank-1 caller
zen4       m0_default                 encrypt_zero_symmetric
express    m0_default                 encrypt_zero_symmetric
zen4       m1_minimal_att_keys        Ciphertext::operator= / resize
express    m1_minimal_att_keys        Ciphertext::resize
bigsmp     m1_rtn_release_early       encrypt_zero_symmetric
```

`patch_minimal_att_keys.py` calls the `create_galois_keys` overload with an explicit step list (one
more frame). Fix: a twelve-frame chain plus `resolve_resident.py --program-root` (first frame under
the program tree, chosen at analysis time).

| caveat | |
|---|---|
| depth clamp | `pmrec` capped `SNNI_PM_DEPTH` at 6 silently while `RUN_META` recorded the request: every "depth 7" table is depth 6. `RUN_META` now carries request, ceiling and effective value. At effective depth 10, 93.8% resolves into two program lines (`off-path-runs/p_depth10/`) |
| naming | early `m0`, `m1` tables named by fixed depth (`encrypt_zero_symmetric`, `mod_switch_scale_to_next`; `Ciphertext::operator=`, `Ciphertext::resize`); same bytes as the chain-keyed tables |

### Checks that reported the opposite of their data

| check | defect | repair |
|---|---|---|
| depth watcher `peek_live.py | grep -i encrypt_zero` | `peek_live.py` prints mapping names; SEAL is static, every row reads `site=test caller=test` | `impl/depth_verdict.sh` symbolises first |
| depth-10 probe showing `MemoryManager::GetPool()` at 87% | `addr2line -i` prints innermost first; `symbolise.resolve(inline=True)` returns the outermost | `impl/peek_deep.sh` prints the whole chain |
| allocated against resident | `peek_live.py` sums allocated bytes (96% touched on the key bank, 90% on `ckks_multiply`) | resident against resident |
| one `m0` re-run, 10.5 h with `attrib_host=off` | direct `sbatch` bypassed `tools/palma/run_step.sh` (sets `CHANNELS`) | |

## Earlier campaign (other instrument; mechanisms only)

| | |
|---|---|
| attention Galois keys reloaded from disk | compiled into the image this campaign did not reuse; a candidate step (reloadable) |
| thread count | 132.7 -> 90.2 -> 73.1 GiB at 72 -> 32 -> 16 threads; 120,605 s at 16 against 54,775 s at 72: paid |
| chunked bootstrapping | 161,807 s against 63,193 s stock: paid (here `m8`) |

## Known off-path in advance

| | |
|---|---|
| lower `remaining_level`, `boot_level`, scale | multiplicative depth and approximation precision: accuracy trade |
| `sec_level_type` | upstream already runs with SEAL's security check disabled (`sec_level_type::none`); memory numbers here are not at a standard security level |
