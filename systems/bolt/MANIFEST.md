# BOLT: manifest

| field | value |
|---|---|
| system | EzPC/SCI + Microsoft SEAL 4.1.1, 2PC hybrid HE + MPC |
| arm | closed: only levers already applied to a primary system; outcome transfer, no-op or regression |
| machine | PALMA `bigsmp` (72-core Skylake AVX-512, 1.5 TB, `--exclusive`); every published row |
| workload | BERT-base, 12 layers, sequence length 128, one SST-2 sample (`id=0`), 2 classes |
| protocol | BOLT 2PC: ALICE (r=1) holds the quantised weights, BOB (r=2) holds the input |
| HE backend | SEAL 4.1.1 (BFV), three contexts, all N=8192: coeff {54,54,55,55} with t=536903681 and t=4295049217, and {60,60,60} with t=557057 |
| MPC backend | SCI: KKOT / split-KKOT oblivious transfer, bitlength 37, `NL_SCALE` 12, `GELU_SCALE` 11 |
| image | `sif/bolt-<step>.sif`, built by `impl/build_sif.sbatch` from `impl/bolt.def` |
| upstream | `github.com/Clive2312/EzPC` branch `bert`, commit `8aaaf32d20b5bdb965989daed8e7d58c39b008b2` |
| binary | `BOLT_BERT`, md5 `b93d8e3e2ebd3d8d3bb62794ab5a8720`, `-march=x86-64-v3 -g` |
| pools | host |
| threads | compile-time `MAX_THREADS` 12 (`linear.h`), 64 (`nonlinear.h`), `NL_NTHREADS` 32 (`bert.h`); `OMP_NUM_THREADS=16` pinned in the sbatch (SEAL, Eigen) |
| published party | higher `VmHWM` of ALICE and BOB; ranking in `party_peaks.txt` |

## Build

```
impl_upstream: https://github.com/Clive2312/EzPC.git (branch bert)
impl_commit: 8aaaf32d20b5bdb965989daed8e7d58c39b008b2
impl_instrument: pin_seed_and_gate.py
impl_build_apptainer: apptainer build bolt-<step_id>.sif impl/bolt-<step_id>.def
impl_build_palma: sbatch --export=ALL,STEP=<step_id> impl/build_sif.sbatch
```

## Target

```
target_host_kb: 135168
overhead_host_kb: measured_per_run
replicates: 3
gate_tier: T2
gate_tolerance_abs: 1.81e-2
gate_observable: BOB's reconstructed class scores, NUM_CLASS = 2 values at full precision, written
                 by the patched binary to logs/gate.txt and announced as
                 `GATE|logit|sample=..|i=..|<value>`; beside it, and NOT part of it, the protocol
                 transcript volume from the program's own `> [NETWORK]` lines in logs/comm.txt
seed: SNNI_SEED=0x5EED, applied at THREE draw sources, each announced separately and each required
      by the runner: `SEEDED|prg128|` (sci::PRG128, the share masks), `SEEDED|prg256|`
      (sci::PRG256, the KKOT oblivious-transfer extension used by every non-linear layer) and
      `SEEDED|seal_prng|` (SEAL's Blake2xb factory, i.e. the secret key, the public key and every
      encryption's noise). Whether that suffices is settled by the two-run check and recorded in README.md rather than asserted here.
```

Units: SEAL BFV, N=8192, 4-prime chain: ciphertext `2 x 4 x 8192 x 8 B = 524,288 B`, batched
plaintext `8192 x 8 B = 65,536 B`. Per layer, with streamed weights:

```
one layer's packed weights   (4 x 768x768 attention + 2 x 768x3072 FFN)
                             = 7,077,888 slots x 8 B                 =  56,623,104 B =  55,296 kB
FFN activation ciphertexts   128 x 3072 / 8192 = 48 CTs x 524,288 B  =  25,165,824 B =  24,576 kB
HE evaluation keys           N=8192 -> 13 power-of-two rotations x 2 directions plus one
                             relinearisation key, at 4 x 2 x 4 x 8192 x 8 B = 2 MiB each
                             = 27 x 2,097,152 B                      =  56,623,104 B =  55,296 kB
                                                                       ----------------------------
                                                                                      135,168 kB
```

| not included | consequence |
|---|---|
| transient per-batch working set of the non-linear layers (GELU: 128 x 3072 = 393,216 elements at bitlength 37) | not resident at the published (ALICE) peak (`steps/s2_threads_4/peak-objects.md` decomposes entirely into HE objects); SCI's KKOT/IKNP extension keeps no persistent OT cache (`SCI/src/OT/ot_pack.h`: only base-OT and block protocol objects, nothing scales with the element count), so there is no Ferret-style persistent buffer to add |
| ALICE's plaintext weight file (held whole upstream) | |

Earlier campaign (other instrument, 16-core EPYC): ALICE 112.6 to 115.4 GiB, BOB 2.2 to 7.0 GiB;
ALICE 7.79 GiB after a streaming-first waterfall.

## Correctness gate

| | |
|---|---|
| observable | BOB's two class scores, `signed_val(...) / 2^NL_SCALE`, printed exactly (`%.17g`) |
| width | 2 values: catches structural changes, not changes confined to a few elements that cancel by pooling |
| companion | transcript bytes and rounds per phase (`> [NETWORK]: ... consumes: N bytes`), exact |
| tier | T2 absolute (ULP per truncation does not scale with the value) |
| replicates | not bit-reproducible: PRG instances derive keys from the seed and a construction counter; construction order varies over 12 to 64 threads |

Tolerance:

| | |
|---|---|
| derivation | 50 `nl.right_shift` operations in `bert.cpp`, at two scales (`NL_SCALE` 12, `GELU_SCALE` 11): 1.81e-2 |
| replicates (`steps/*/logs/run*/gate.txt`) | `s0` pairs 1.05e-2, 1.22e-3, 9.52e-3; every `s1`/`s2` run within 1.46e-2 of every `s0` run |
| validation (`x1_layer1`, off-path) | one-layer rebuild, 6 truncating operations, disagreement 3.42e-3, about a third of 1.05e-2 for 8.3x fewer operations; an unpinned draw source would not scale with depth |

## Machine type: `bigsmp`

`x_bigsmp_s2`: the `s1` image at `OMP_NUM_THREADS=4` (`s2`'s configuration), 3 replicates on each
partition. `long`: 36 cores / 192 GB; `bigsmp`: 72 cores / 1.5 TB; both AVX-512 Skylake.

| step | | `long` | `bigsmp` | delta |
|---|---|---:|---:|---:|
| `x_bigsmp_s2` (4 threads, 17 GB) | peak median | 17,142,140 | 17,155,324 | +0.077% |
| | own spread | 0.27% | 0.81% | |
| | wall median | 799 s | 768 s | -3.9% |
| `s0_default` (16 threads, 107 GB) | peak median | 112,809,840 | 108,852,060 | -3.5% |
| | own spread | 8.90% | 3.16% | |
| | wall median | 398 s | 487 s | +22.4% |

- The peak transfers (the baseline's -3.5% is inside its 8.90% spread on `long`).
- Wall does not: -3.9% on the small step, +22.4% on the baseline (a 107 GiB working set at 16
  threads spans more memory controllers on the larger node).
- `runtime_delta_pct` is step against predecessor on one machine; pre-move runtimes are not
  comparable and were superseded by the re-measurement at depth 5.
- Reason for the move: `long` had 50 of 90 nodes allocated, start estimated four days out; `bigsmp`
  had an idle node.
