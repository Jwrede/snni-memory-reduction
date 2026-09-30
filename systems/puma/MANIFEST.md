# PUMA: manifest

| field | value |
|---|---|
| system | SecretFlow SPU, ABY3 3PC |
| arm | closed: levers are borrowed from primary systems; outcome per lever is transfer, no-op or regression |
| machine | PALMA `express` (36-core Skylake AVX-512, 95 GB, `--exclusive`) |
| workload | BERT-base uncased, 12 layers, sequence length 128, one sample |
| protocol | ABY3, FM64, 3PC on loopback: 3 SPU nodes + 2 PYU nodes + the driver |
| image | `sif/puma-<step>.sif`, built by `impl/build_sif.sbatch` from `impl/puma.def` |
| stack | spu 0.9.5, jax 0.4.34, flax 0.8.5, transformers 4.45.2, numpy<2 |
| spu examples commit | `02ac612de3084dc53eddab91b720c6b80bc48779` |
| measured program | `impl/snni_bert_puma.py`, bind-mounted; md5 in every RUN_META |
| pools | host |
| threads | `OMP_NUM_THREADS=8` pinned in the sbatch (upstream default); SPU `max_concurrency` at its default, declared |
| published process | the highest `VmHWM` of the six polled processes; ranking in `party_peaks.txt` |

## Build

```
impl_upstream: https://github.com/secretflow/spu.git (sparse: examples)
impl_commit: 02ac612de3084dc53eddab91b720c6b80bc48779
impl_build_apptainer: apptainer build puma-<step_id>.sif impl/puma.def
impl_build_palma: sbatch --export=ALL,STEP=<step_id> impl/build_sif.sbatch
```

The measured program is the driver (one Python file); a step selects it with `PUMA_DRIVER` and
declares a snapshot.

## Target

```
target_host_kb: 366264
overhead_host_kb: measured_per_run
replicates: 3
gate_tier: T2
gate_tolerance_abs: 1.4e-2
gate_observable: the encoder's LAST hidden state, 128 x 768 = 98304 values, written by the
                 measured driver to logs/gate.txt one value per line and announced as
                 `GATE|hidden_last|n=98304|sum=..|absmax=..`; plus `GATE|logits|` and the
                 plaintext reference `REFERENCE|cpu_logits|` as evidence beside it, and
                 `GATE_NONZERO|n/98304` so a degenerate observable fails instead of passing
seed: SNNI_SEED=0x5EED, applied in the measured driver to numpy's global generator, to the JAX
      PRNG key that from_pretrained uses for the freshly initialised classification head, and to
      SPU's RuntimeConfig.public_random_seed, announced as
      `SEEDED|numpy+jax+spu_public|seed=0x5EED`; the run fails without that marker. The SECRET
      SHARES are not pinned and cannot be pinned from here (SecureRandSeed inside the SPU
      runtime, no configuration knob), so whether this system reproduces is settled by the
      two-run check and recorded in README.md rather than asserted here.
```

Target: the ABY3-replicated word-embedding table on P2:

```
2 * V * d * 8 = 2 * 30522 * 768 * 8 = 375,054,336 B = 366,264 KiB
```

`overhead_host_kb`: file-backed total of each run's own peak smaps.

## Gate

| | |
|---|---|
| observable | encoder's last hidden state; `bert-base-uncased` has no fine-tuned head, so the logits are a random projection (emitted as evidence, with the plaintext reference) |
| tolerance | absolute: a fixed-point ULP per truncation does not scale with the value |
| derivation | written before any comparison, below |
| measured 2026-08-02 | worst absolute difference over all replicate pairs 3.33e-3 (4.2x inside); mean 1.4e-4 to 2.9e-4 against a median magnitude 0.178; worst relative 2.7e8 (values near zero) |
| replicates | not byte-identical (unpinnable shares), so T2 |
| degenerate check | `GATE_NONZERO|`; the harness fails on `GATE_DEGENERATE` |

```
fxp fraction bits, FM64                      18      -> 1 ULP = 2^-18 = 3.8e-6
sequential truncations on the deepest path   ~30 per layer x 12 layers = 360
layer gain (activation magnitude growth)     ~10
                     360 x 3.8e-6 x 10  =  1.4e-2 absolute
```

## Threads

| knob | earlier campaign result |
|---|---|
| `OMP_NUM_THREADS` | `02_thread` set it to 1: no-op (OMP does not bind SPU's scheduler) |
| SPU `max_concurrency` | `08_maxconc`: no-op (protocol-round-bound) |

Both pinned (thread count changes per-thread buffers and arenas).

## Processes

| process | role |
|---|---|
| 3 SPU nodes | protocol parties |
| PYU P1 | holds the input |
| PYU P2 | holds the plaintext model |
| driver | loads the model, hands it to P2, reconstructs the result; holds a plaintext copy (not part of a deployment, polled anyway) |

The per-process maximum is published, never the sum (earlier campaign: 6.60 GiB per party, 18.99 GiB
summed over the three ABY3 parties).
