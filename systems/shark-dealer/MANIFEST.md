# SHARK-DEALER: manifest

| | |
|---|---|
| subject | SHARK's offline dealer (party 2), which generates all correlated randomness before the online parties run |
| separate system because | different process, floor and gate; per-role convention (as LLAMA's dealer) |
| binary | same images, upstream commit and binary as the online line (`sif/shark-<step>.sif`, `github.com/kanav99/shark` at `6957e5ed7d279d2cd9fac0b4c1a6f30a77f91bc1`); a different process is polled |
| pools | host |

## Build

```
impl_base_image: the images SHARK's own steps run; see systems/shark/MANIFEST.md
impl_artifacts_from: shark
impl_instrument: patches/socket_full_transfer.diff
impl_build_docker: docker build -f ../shark/impl/Dockerfile.step --build-arg SHARK_SRC=<file> ../shark
impl_build_palma: sbatch impl/../../shark/impl/build_sif.sbatch
```

`d1_layer_weights` runs the online image `shark-s2_layer_weights.sif` of the T=4 order (weights plus
batch check), linked under the dealer step's name.

## Hardware

| | |
|---|---|
| machine | PALMA `zen2-128C-496G`, `--exclusive`: AMD EPYC 7742, 128 cores, 496 GB |
| threads | `OMP_NUM_THREADS=16` (the online line's operating point; recorded in every dealer run) |
| allocator | glibc default, not pinned |
| key scratch | `/scratch/tmp`, ~112 GB written per run, deleted after the run |

## What is measured

| phase | measured |
|---|---|
| 1: dealer writes `server.dat`, `client.dat` | polled, frozen, decomposed |
| 2: online parties consume the files | run, not polled; confirms the key material works |

## Target

```
target_host_kb: 181248
overhead_host_kb: measured_per_run
replicates: 3
gate_tier: T1
gate_observable: the md5 of both preprocessing files (`keys.md5`), which IS this role's output.
                 The online line deliberately does not hash them -- minutes per replicate, and the
                 online output is what an online step must not change -- but for the dealer they
                 are the only observable that means anything, so this line pays for them.
seed: the dealer's PRNG is pinned in the library: `init::gen` seeds `prngGlobal` with the constant
      `0xdeadbeef` (`init.cpp:20`). The measured binary additionally calls `srand(0x5EED)` and
      emits a `SEEDED|` marker, which is what the online parties need; the dealer does not depend
      on it. Determinism is therefore expected to be BYTE-EXACT here, which is what makes T1
      possible: the same check on the LLAMA dealer produced 47 GB of byte-identical key material
      across every run of every step.
```

Per-element key layout as the online target (2,196 B per element at `f = 16`;
`systems/shark/MANIFEST.md`). The dealer builds both parties' keys at once (`KeyGenMatMul`,
`conv.cpp:49`: `k0.{a,b,c}`, `k1.{a,b,c}` and a temporary) and writes each party's keys as it
produces them, so one batch's generation state plus write buffers must be resident:

```
generation state for one ARS batch (98,304 elements, both parties)   ~ 2 x 84,672 kB = 169,344 kB
buffered writes to two files                                          ~     11,904 kB
                                                                       -----------------
                                                                              181,248 kB
```

- Provisional term: the coexistence of the two parties' batches is read from `conv.cpp`, not
  measured.
- Cross-check against the earlier campaign:

| | earlier campaign | this campaign |
|---|---:|---:|
| baseline | 887 MB | 867 MB (`d0_default`, n=3, spread 0.098%) |
| after optimisation | 209 MB | 258 MB after one borrowed free step (`d1_layer_weights`) |

## Purpose

The LLAMA dealer (`d0`-`d7`, 7.51x) and this dealer are the two FSS dealers measured the same way;
they differ in how much of the dealer's work is resident at once.
