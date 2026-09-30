# BumbleBee: manifest

| | |
|---|---|
| paradigm | hybrid HE + MPC: Cheetah 2PC over SPU; lattice HE (Microsoft SEAL, CKKS-shaped parameters as a plain RLWE ring) for linear layers, secret sharing plus Ferret OT for non-linear ones |
| model | BERT-base shapes, 12 layers, hidden 768, 12 heads, FFN 3072, sequence length 128 |
| ring | FM64, 16 fractional bits |
| pools | host |
| upstream | `https://github.com/AntCPLab/OpenBumbleBee` at `47c5560d069543591f3b0176eb530ede28e33fc3` |
| paper | Lu et al., *BumbleBee: Secure Two-party Inference Framework for Large Transformers*, NDSS 2025 |
| arm | open (primary system) |

## Build

```
impl_upstream: https://github.com/AntCPLab/OpenBumbleBee.git
impl_commit: 47c5560d069543591f3b0176eb530ede28e33fc3
impl_build_palma: sbatch --export=ALL,STEP=<step_id> impl/build_step.sbatch
```

- Each source step carries `impl/patches/<lever>.diff` and an `impl/tree/` with the complete changed
  files; a step is one overlay directory mirroring the OpenBumbleBee tree.
- `b5`-`b8` change no source; they set `BB_MAX_CONCURRENCY` in `levers.env`.
- The binary md5 of each step is recorded and compared with its parent's at build time.

| file | content |
|---|---|
| `impl/src/op_bench.cc` | the measured program: the `00_default` benchmark plus the gate |
| `impl/src/BUILD.bazel` | its bazel target, plus the `public_helper` dependency of the gate |
| `impl/patches/` | the same change as a diff against the pristine file |
| `impl/build_step.sbatch` | one step, one image, built on the cluster |
| `impl/bumblebee.sbatch` | the measurement job |
| `impl/run_bumblebee.sh` | inner launcher: two parties, party pids, gate extraction |
| `impl/derive.sbatch` | raw run -> object tables, per party |

## Hardware

| | |
|---|---|
| machine | PALMA `express`, `--exclusive`: 36-core Skylake AVX-512, 95 GB (same node type as `normal`; shorter queue) |
| parties | 2 on one node, brpc over loopback, both measured |
| `OMP_NUM_THREADS` | 16 per party, pinned (2 x 16 = 32 of 36 cores; yacl `parallel_for`) |
| `BB_MAX_CONCURRENCY` | 16 at the baseline; the lever of `b5`-`b8` |

`BB_MAX_CONCURRENCY` reaches `conf.set_max_concurrency()` and
`ctx->getClusterLevelMaxConcurrency()` (`libspu/mpc/cheetah/protocol.cc:52`) and sizes only
`CheetahOTState`:

| quantity | value | source |
|---|---|---|
| OT instances | `min(48, cluster_level_max_concurrency)` | `libspu/mpc/cheetah/state.h` |
| per instance | one Ferret sender and one receiver adapter | `libspu/mpc/cheetah/ot/basic_ot_prot.cc:28-46` |
| per adapter | `lpn_param_.n * sizeof(uint128_t)` = 10485760 x 16 B = 160 MiB | `libspu/mpc/cheetah/ot/yacl/yacl_ote_adapter.h:58,114` |
| at 16 instances | 16 x 320 MiB = 5.0 GiB | |

On the `b5`-`b8` rows: `OMP_NUM_THREADS` stays 16 (unlike the earlier campaign's `s4_t1`, which moved
both); the transcript changes by 0.62 to 0.74 MB per instance removed (one-time Ferret setup; run-to-run
spread ~2e-6); the floor is one instance.

## Workload

Benchmark with BERT-base shapes and operator mix on dummy weights, without the embedding, written for
this study on SPU's C++ interface (`impl/src/op_bench.cc`; not part of the upstream release).

| | |
|---|---|
| non-linear operators | faithful, applied to every step's tree by `impl/patches/faithful_nonlinear.py` |
| `softmax_row()` | row maximum (halving reduction with `hal::max`), `x - max`, `f_neg_exp_taylor`, row sum, reciprocal of the row sum, broadcast multiply; scores scaled by `1/sqrt(64) = 0.125` |
| `layernorm()` | row mean, variance, `hal::rsqrt(var + 1e-5)`, gamma, beta |
| weights | dummy, secret-shared like the input (`hal::seal`): every matmul has two shared operands; BumbleBee's deployment (server weights in the clear) is not measured |
| left out | embedding, pooler/classifier head, real weight values |

The stock benchmark's placeholder softmax and affine LayerNorm gave peaks within the replicate spread
of this line (b0 15,460,296 against 15,480,820 kB, b8 1,478,276 against 1,460,632 kB); the faithful
softmax sends 39% fewer bytes (one reciprocal per row instead of per score).

## Target

```
target_host_kb: 406016
replicates: 3
gate_tier: T2
gate_tolerance_abs: 7.3e-3
gate_observable: the reconstructed output of the FIRST transformer layer, 128 x 768 values, written
                 by rank 0 to logs/gate.txt one value per line and announced as
                 `GATE|layer1|rank=0|n=98304|file=...`; plus per-party checksums `GATESUM|layer1|`
                 and `GATESUM|final|`; plus the protocol transcript volume
                 `COMM|rank=N|sent=..|recv=..` in logs/comm.txt
seed: no call to seed anything exists in the measured program; every plaintext comes from
      std::mt19937 with seeds fixed in the source (op_bench.cc rand_floats/make_secret, weights
      100..1200, input 9999), announced at startup as `SEEDED|mt19937|weights=100..1200|input=9999`.
      The protocol randomness (secret shares, HE masks, Ferret OT) is drawn inside SPU and yacl and
      is NOT pinned here; whether it reproduces is settled empirically by the two-run check, not by
      inference.
overhead_host_kb: measured_per_run
```

One OT instance plus the heaviest single linear operation:

| term | bytes | content |
|---|---:|---|
| Ferret OT preprocessing | 335,544,320 | one OT instance: one sender and one receiver adapter, each `lpn_param_.n * 16 B` |
| HE working set | 80,216,064 | FFN up-projection `(128 x 768) x (768 x 3072)`, all three operands at the 3-prime encoding level |
| total | 415,760,384 | = 406,016 kB |
| measured validation | | OT +2.0 to 2.2%, HE +0.9 to 1.1%, sum +1.8 to 2.0% over 415.8 MB in 3 runs; stock encoding 8.47 GB for one matmul (`off-path-runs/p_floor_bench/`) |

At 16 instances, 15/16 of the measured OT term is above the floor.

## Correctness gate

| | |
|---|---|
| tier | T2, absolute tolerance 7.3e-3 (Cheetah is not bit-exact by design; T1 is unavailable) |
| tolerance derivation | `32 sequential multiplications x 2^-16 x layer gain 15 = 7.3e-3`, written before any comparison |
| why absolute | error is one truncation ULP per multiplication; three near-zero elements give a relative difference of 16.3 between two runs of one binary, absolute 5.1e-3 |
| observable | first-layer output, rank 0, full tensor; both parties emit checksums |
| stock program | revealed the output and dropped it; `op_bench.cc` prints it (`impl/patches/`, applied to every step) |
| measured | worst b0-b0 difference 4.62e-3; worst step-against-b0 5.58e-3 (`b5`) |
| `final` checksum | evidence beside the gate: with the faithful operators the 12-layer output does not wrap (rank-0 `final` of all 27 runs between -4704.9 and -4700.9) |

## Instrument defects found here

| defect | effect | handling |
|---|---|---|
| `tools/symbolise.py` offset formula assumes `p_vaddr == p_offset`; this binary's text segment is padded | every address one page low; a libgcc unwinder symbol published as the 6.1 GB top object | reported; worked around in `impl/derive.sbatch` |
| SPU's `.bazelrc` links the C++ runtime statically | `operator new` internal, not interposable by `LD_PRELOAD`; 6.10 of 6.66 GB on one site | the measured build overrides the two link settings; `impl/build_step.sbatch` fails if they revert |
| recorder did not see SEAL's allocations | | fixed campaign-wide |
