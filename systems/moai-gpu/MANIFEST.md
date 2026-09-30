# MOAI-GPU: manifest

| | |
|---|---|
| paradigm | pure FHE, RNS-CKKS on PhantomFHE, single server process, non-interactive |
| workload | one BERT-base encoder layer, batch 256, seq 128 (scope below) |
| pools | VRAM and host |
| upstream | `github.com/dtc2025ag/MOAI_GPU`, pristine commit `b780f62` (= `origin/main`, the root commit) |
| branch | `waterfall-final` @ `d2792f3` (checkout outside this repository); `main` (also ours) forks from the step line at `da266b5` |
| baseline | `b780f62` plus seed pinning and gate emitter (`impl/pin_seed_and_gate.py`), applied to every step |
| cluster workdir | `/scratch/tmp/j_wred02/snni_campaign/moai-gpu` (earlier campaign: `/scratch/tmp/j_wred02/moai_gpu`, 522 GB) |

| decision | record |
|---|---|
| ciphertext levels predict the measured VRAM | `level-audit.md` |
| dynamic cudart indistinguishable from static | `linkmode.md` |
| device attribution | `tools/attrib_device/cudarec.so`; table is a named file-backed mapping (`SNNI_CUDAREC_TABLE=devtable_<pid>`, since 2026-08-21), labelled `instrument:` and subtracted |

## Build

```
impl_base_image: the MOAI-GPU baseline image; impl/extract_src.sbatch lifts /moai out of it
impl_build_palma: sbatch --export=ALL,STEP=<step_id>,SRC=<prepared tree> impl/build_step_image.sbatch
```

- `impl/extract_src.sbatch` lifts `/moai` from the baseline image; step patches apply to that tree.
  Prepared trees (32 GB) are not published; the patches are.
- Each prepared tree was diffed against the baseline: `fft.h`/`ckks.cu` at `s1`,
  `+test_single_layer.cuh` at `s2`, `+include.cuh` at `s5`, `+layernorm.cuh` at `s6`,
  `+single_att_block.cuh` at `s13`. `s8` carries `patch_stream_boot1_ln1.py` alone
  (`wpark_helpers.cuh` unchanged at `s8`).
- `s1`'s source change has no artifact in `impl/` (`# impl_lever: unpublished`); the prepared tree of
  `s12_ffn_threads` was not retained under that name (declaration rests on the patch name).

## Hardware

| | |
|---|---|
| cluster | PALMA, partition `gpuh200` |
| device | one full NVIDIA H200, 141 GB, `--gres=gpu:1` (pristine baseline reserved 129.16 GiB in the earlier campaign) |
| host | 16 cores, 110 GB requested |

## Scope: one layer

- The peak sits inside a bootstrap; every layer runs the same bootstraps against the same rotation-key
  bank; a second layer repeats the shape.
- No twelve-layer runtime claim; the peak is per batch of 256, not per inference.
- The earlier campaign's BERT 72.3 GiB against ViT 73.8 GiB on CPU is not evidence of
  sequence-length insensitivity: ViT's 197 tokens were padded to 256 and the batch halved
  (`num_X 256 x num_row 128` and `num_X 128 x num_row 256`, both 32,768 slots); dummy weights; no ViT
  baseline (`SNNI-Benchmark/canonical_runs/run_moai_vit_2layer.sh`).

## Targets

```
target_vram_kb: 25267200
target_host_kb: 18432
overhead_vram_kb: 0
overhead_host_kb: measured_per_run
```

| pool | target | derivation |
|---|---|---|
| VRAM | 25,267,200 kB | MOAI-CPU's streamed target (same protocol, model and CKKS parameters: N = 2^16, same modulus chain, key-switching key 1,321,205,760 B on both, bootstrap outputs at level 20, batch 256): bootstrap instant at a chunk of 128 ciphertexts (largest boot key stage, chunk in flight, relinearization key, one thread's working set); `systems/moai-cpu/MANIFEST.md` |
| host | 18,432 kB | one FFN weight matrix, 768 x 3072 x 8 B = 18,874,368 B |

- `overhead_vram_kb` 0: the device peak is Phantom's pool reserved high-water (`snni_gpupeak`); the
  CUDA context (~650 MiB per process) is outside it.
- Host overhead: file-backed total of each run's peak smaps.

## Correctness gate

```
gate_tier: T1s
gate_observable: GATE|out|n=..|sum=..|sumsq=..|absmax=.. over the 768 output ciphertexts' b_vec slots, plus GATE|ct0[i]= for the first eight, printed at 17 significant digits by the measured binary
seed: SNNI_SEED (default 0x5EED) driving BOTH draw sources: phantom-fhe random_bytes (prng.cuh), which fixes the key material's values, and adjust_sk_hamming_weight (secretkey.cu:376), which fixes WHICH secret-key coefficients stay non-zero; emits SEEDED|phantom_random_bytes| and SEEDED|phantom_sk_hamming|
replicates: 1
```

| | |
|---|---|
| gated quantity | modulus-chain trace at 16 phase boundaries: byte-identical over three runs and both seeds (md5 `7d2520932ba9`) |
| admission | discrimination test passed in both directions (`off-path-runs/p_level_discriminate`, `p_level_discriminate2`): one post-bootstrap mod-switch removed shifts downstream levels up by one, one added shifts them down, from the first reachable line |
| not covered | changes confined to values |
| `GATE|out` | printed as evidence |

Determinism, measured:

| | |
|---|---|
| key material | bit-identical across runs |
| structure (modulus chain) | deterministic across runs and seeds |
| values | nondeterministic; no seed changes it |

- Pristine test asserted nothing (decrypt checks commented out; `single layer test passed!` printed
  unconditionally). The observable comes from the earlier campaign's `FP|` block
  (`test_single_layer.cuh:1346`).
- Seed, first attempt: `random_bytes()` (`secretkey.cu:355-356`) only. Two-run check (jobs
  45633998/99): `sum=16731.438430549471` against `sum=24932.468420197478`, `ct0[0]` -144.4 against
  -319.4, `sumsq` within 0.8%.
- Second source: `adjust_sk_hamming_weight` (`secretkey.cu:376`), shuffling with
  `std::mt19937 gen(std::random_device{}())`. Both seeded; `build_step_image.sbatch` checks each marker.
- Not reaching the output: `util.cuh:generator()` (only via `my_rand_int`, never called); `numth.cu`
  Miller-Rabin witnesses and a primitive root normalised by `try_minimal_primitive_root`.
- With both seeded (jobs 45710206/07):

```
run1  KEYFP|ct0|n=1966080|fnv=6677111189981983917    GATE|out|sum=-108693.68
run2  KEYFP|ct0|n=1966080|fnv=6677111189981983917    GATE|out|sum= -31816.20
```

  Keys and encryption randomness identical, output not: the divergence arises in the computation
  (non-deterministic GPU reduction order, amplified by arithmetic already known to be wrong: LayerNorm
  variances of -3.1e7, `enc_softmax` decoding to ~1e31). A seed at a different value lies no further
  from either run than they lie from each other.

## Replicates

`replicates: 1` (`tools/palma/conf/moai-gpu.conf`): a run is 56 minutes; three replicates across three
channels were nine GPU-hours per step on a partition with two live nodes. `cost_type` is declared per
step in `levers.env`.
