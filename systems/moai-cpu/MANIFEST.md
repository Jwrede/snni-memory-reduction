# MOAI-CPU: manifest

| field | value |
|---|---|
| system | MOAI on SEAL 4.1 with a bootstrapping fork, non-interactive CKKS; primary system of the pure-FHE paradigm on CPU; ARION's partner (same declared depth) |
| machine | PALMA `zen4` (192-core AMD Zen 4 AVX-512, 740 GB), non-exclusive, fixed 16-core cpuset on one NUMA node (an exclusive node for the whole line exceeded fair share and queue time). Sharing leaves the peak unchanged and inflates bandwidth-bound phases (bootstraps up to +53%, layer norm +52%) |
| machine move | off `bigsmp` by measurement: same image, workload and threads peak at 402,242,084 kB here against 400,914,264 kB on `bigsmp` (+0.33%); 46 nodes instead of three. `bigsmp` and earlier exclusive runs are history |
| workload | BERT-base, num_X=256, num_row=128, num_col=768, two of twelve layers, declared |
| scheme | CKKS, logN=16 (N=65536), logp=46, logq=51, special prime 58, `remaining_level` 20, `boot_level` 14, secret-key Hamming weight 192, `sec_level_type::none` |
| image | `sif/moai-cpu-<step>.sif`, built by `impl/build_sif.sbatch` from `impl/moai-cpu.def` |
| upstream | `github.com/dtc2025ag/MOAI`; commit also in `/opt/moai_commit.txt` in the image |
| pools | host only, single process |
| recorder depth | `SNNI_PM_DEPTH=7` (`impl/moai_cpu_campaign.sbatch`, recorded in `RUN_META`); depth 5 lands on SEAL's `Ciphertext::operator=` (55.4%) and `Ciphertext::resize` (37.2%); depth 7 reaches program lines (`Ct_pt_matrix_mul.hpp:87` 8.1%, `test_full_scheme.hpp:522` 7.1%, `layernorm.hpp:320` 4.0%). Tables are resolved with `--program-root /root/moai` |
| replicates | 1 (since 2026-08-05): a replicate took 22 h 45 min on `bigsmp`, 10 h 40 min on `zen4`. Reasoning: `tools/palma/conf/moai-cpu.conf`; every row carries `n_runs=1` |
| threads | `OMP_NUM_THREADS=16` (campaign decision of 2026-08-03; the program has no thread default, the earlier 72 was the node's core count) |

## Declarations

```
target_host_kb: 25267200
overhead_host_kb: measured_per_run
replicates: 1
spread_host_pct_declared: 2.0    (run-to-run spread of a PHASE maximum on the shared-node regime, from five identical-code phase pairs of the re-measurement line, off-path.md "Phase-level spread on shared nodes"; used by the inert-step guard at n=1)
gate_tier: T2
gate_tolerance: 1e-6
transcript_none: single process, non-interactive CKKS; no party exchanges bytes, so phase B rests on the plaintext test (PROCEDURE, "Two phases of evidence")
gate_observable: the decrypted per-layer output the PRISTINE program already writes, layer_<id>.txt
                 in its working directory, flattened to one value per line into logs/gate.txt and
                 announced as `GATE|layer_outputs|files=N|n=..` plus `GATE_NONZERO|nz/n`
seed: SNNI_SEED=0x5EED, applied to SEAL's PRNG factory (Blake2xbPRNGFactory) before the SEALContext
      is built, which is where the secret key, the public key, the relinearisation and Galois keys
      and every encryption's noise are drawn; announced as `SEEDED|seal_prng|seed=0x5EED` and the
      run fails without the marker. There is exactly ONE KeyGenerator in the measured path
      (test_full_scheme.hpp), and the patch script prints every other draw source it did not cover.
```

## Build

```
impl_upstream: https://github.com/dtc2025ag/MOAI.git
impl_commit: 8bcb0eadac299db88ca6417c5710cd65ada66af1
impl_instrument: pin_seed_and_gate.py
impl_build_apptainer: apptainer build moai-cpu-<step_id>.sif impl/moai-cpu-<step_id>.def
impl_build_palma: sbatch --export=ALL,STEP=<step_id> impl/build_sif.sbatch
```

## Target

Rule: maximum over a layer's instants
once every streamable term is streamed, at the granularity the line processes at once. From `m8` the
bootstrap runs on chunks of 128 ciphertexts (`BOOT_CHUNK_SIZE`, `patch_chunked_bootstrap.py`; stock:
all 768), so the target holds that chunk in flight. One thread; SEAL context excluded (overhead).

| unit | value |
|---|---|
| ciphertext at chain index c | `(c+1)` MiB (`2 polys x (c+1) limbs x 65536 x 8 B`) |
| chain indices (program's `stdout.log`) | before attention 14, attention result 1, after selfoutput 0, after every bootstrap 20, FFN input 9, intermediate 8 |
| per head | col_W = 64: Q, K, V 64 ciphertexts, scores 128 |
| key-switching key | 1,290,240 kB (35 decomp x 2 polys x 36 limbs x 65536 x 8 B); resident at that size (`off-path-runs/p_floor_bench/`, job 46667236: largest boot stage 19,361,400 kB against 19,353,600 for 15 keys) |
| measured validation | bootstrap instant built as one set 25,555,412 kB, +1.14% over 25,267,200; 15 stage keys within 0.0005%; 3 runs (`off-path-runs/p_floor_bench_boot/`) |

| instant | live under streaming | kB | GB |
|---|---|---|---|
| QK^T in a head | 13 distinct attention rotation keys 16,773,120 + Q 64@13 917,504 + K 917,504 + one thread's rotated weight copy 917,504 + scores 128@12 1,703,936 + V 64@2 196,608 + relin 1,290,240 | 22,716,416 | 23.3 |
| **bootstrap, chunk 128** | boot key stage 15 keys 19,353,600 + chunk in flight 128 ct at 32 limbs 4,194,304 + relin 1,290,240 + thread 429,056 | **25,267,200** | **25.9** |
| bootstrap after attention, before LN2 (chunk 1) | boot key stage 15 keys 19,353,600 + output ct 21,504 + residual ct 21,504 + relin 1,290,240 + thread 429,056 | 21,115,904 | 21.6 |
| softmax bootstrap in a head | boot key stage 19,353,600 + exp(x) block ~1,075,000 + the one ct bootstrapped 21,504 + V 196,608 + relin 1,290,240 | ~21,940,000 | ~22.4 |
| bootstrap after LN1 (chunk 1) | boot key stage 19,353,600 + output ct + relin + thread | ~21,100,000 | 21.6 |
| feed-forward, column by column | input chunk 128@9 1,376,256 + output accumulator 768@0 786,432 + relin | ~3,400,000 | 3.4 |

- Maximum: bootstrap at chunk 128 (the endpoint table measures the chunk row at 4,295.5 MB). Input
  spilled (m21), output written chunkwise (m8), residual copy parked (m25): no full hidden-state vector.
- Attention keys: the attention requests 14 of the 31 default rotation steps (`m2_minimal_att_keys`).
  At 32,768 slots +16,384 and -16,384 map to the same Galois element (the order of 3 modulo 2N is N/2):
  the 31 default steps give the 30 keys the stock program builds; the 14 requested steps give 13
  distinct keys. The m1->m2 drop of 22.3 GB matches 17 keys removed (17 x 1.32 GB = 22.5 GB). Step
  files and patches say "14 attention rotation keys" for the 14 requested steps.
- Streamed, not counted: layer input X (m24), the 47-key bootstrap bank (one BSGS stage at a time, m7),
  full hidden-state vectors (m8, m21, m25), outputs of already-computed heads (768 ciphertexts at chain
  1, 1,572,864 kB = 1.6 GB). Counting the head outputs raises the last head's QK^T to 24.9 GB, below the
  chunk-128 bootstrap.
- Finer chunking (`m27`, chunk 64) goes below the target's granularity: off-path (README.md rule 6).
- Per-thread scratch beyond one thread is a concurrency cost, excluded. At T=16 the bootstrap's
  key-switch scratch is 16 x (35 x 36 + 2) x 65536 x 8 B = 10,911,744 kB derived (16.9 GB measured over
  the endpoint's polynomial-evaluation sites): configuration floor ~35.5 GB (endpoint discussion only).

```
coefficient chain: 1 x logq(51) + 20 x logp(46) + 14 x logq(51) + 1 x special(58) = 36 primes
one key-switching key = 35 decomp x 2 polys x 36 limbs x 65536 coeffs x 8 B = 1,321,205,760 B
```

Measured key counts (`off-path-runs/p_keyaudit`, job 45846530): `gal_keys` 30, `gal_keys_boot` 47,
`relin_keys` 1; per key 1,321,227,193 and 1,321,220,871 B serialised (0.0016% above the derivation).


## Host producer: the SEAL pool

`pmrec` hooks `malloc`/`free`; SEAL's pool sits between the program and `malloc` (growth visible,
returns not). Second producer: `tools/attrib_seal_pool/` (adapter contract, README.md).

At `m0`'s peak:

```
pmrec, pool growth      227.9 GB in    373 items   Ct_pt_matrix_mul.hpp:35   (chunks)
the pool recorder       234.9 GB in 10,818 items   gelu_others.hpp:143       (ciphertexts)
```

| | |
|---|---|
| published table | the live set (pool recorder); pmrec's beside it (`objects_host_pmrec_*.jsonl`) |
| rule 4 | ranked against the live table; a step whose target only pmrec sees declares every live row above it skipped |
| retention row | `sealpool:retained` (since 2026-08-28): at `m0`'s peak 239.1 GB live and 172.8 GB held by the pool for released objects; pmrec puts 411.0 GB on SEAL paths and 20.0 MB elsewhere; `unattributed_host` 0.0002 |
| retention as a target | only for a step declaring `runtime_knob: yes` (as `go:runtime_retained`); `derive.sbatch` passes `--retention-row` for this system only |
| names | `--program-root /root/moai`; the recording pool inserts two frames, so fixed-index names shift (`off-path-runs/p_prm0/logs/run1/crosswalk.out`) |
| cost | peak 402,401,204 against 402,436,136 kB (-0.0087%, gate bit-identical); wall 38,747 against 38,283 s (+1.21%); table subtracted as `pooltable_` |

## Thread count, a declared operating point

The program ships no thread default; ARION hardcodes 64 and keeps it.

| | peak | wall |
|---:|---:|---:|
| T=72 | 629,369,264 kB = 600.2 GiB | 13 h 20 min |
| T=16 | 402,690,024 kB = 384.1 GiB | 22 h 45 min |
| | -36.0% | +70.6% |

Not a step (no shipped default to reduce from). T=72 run of the same binary kept in
`results/.superseded-m0_default-t72-*`. Earlier campaign: 132.7 -> 90.2 -> 73.1 GiB at 72 -> 32 -> 16
threads.

## Two layers, declared

- ARION cannot run deeper on the same hardware; the earlier campaign measured the stock two-layer run
  at 584 GiB and 17.6 h (twelve layers: over four days per replicate).
- Source default stays 12; the sbatch sets the depth; recorded in RUN_META and announced as
  `WORKLOAD|model=bert-base|layers=2|of=12|...`.

## Gate

- Observable: upstream's own `layer_<id>.txt` (decrypted per-layer output); nothing added. An earlier
  patch draft added a decryption of the final 768 ciphertexts and was removed.
- `gate_tolerance: 1e-6` relative (CKKS noise rides on the scale 2^46).
- Working directory: the runner creates a writable directory in the results tree with symlinks to the
  image's read-only weight data (earlier campaign: `--writable-tmpfs`).
