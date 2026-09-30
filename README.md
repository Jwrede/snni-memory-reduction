# SNNI memory reduction

This repository accompanies the master's thesis *Optimizing the Memory Footprint of Secure Neural
Network Inference for Transformer Architectures* (University of Münster). It holds the measurement
harness, the reduction procedure, and every measured step of the systems the thesis reports, with
the raw logs each published number is regenerated from.

Each system is reduced as a line, a sequence of steps from the unmodified baseline to an endpoint.
A step changes one thing, is measured on a fixed machine, and is published as one row of the
system's `steps.csv`. The peak of each pool is decomposed into the objects that hold it, and the
waste factor `W` gives the peak, less the loaded-library overhead, in multiples of a
protocol-derived target. This file states the procedure every step follows, how memory is
measured, and how a new system is brought onto the harness.

## Systems

Every line runs BERT-base at sequence length 128, and the two transfer lines run ViT-Base/16 on
SHAFT. The reduction is baseline peak over endpoint peak. `n` is the number of runs per step; the four
systems with one run per step give their reason in their `MANIFEST.md`.

| system | paradigm | steps | reduction | `W` at the endpoint | `n` |
|---|---|---|---|---|---|
| SHARK | FSS | s0-s5 | 123.6x host | 2.19 | 3 |
| SHARK dealer | FSS, offline dealer | d0-d1 | 3.4x host | 1.40 | 3 |
| LLAMA | FSS | o0-o3 | 25.8x host | 36.5 | 3 |
| LLAMA dealer | FSS, offline dealer | d0-d7 | 7.5x host | 8.73 | 3 |
| SHAFT-CPU | arithmetic secret sharing | s0-s8 | 4.3x host | 3.30 | 3 |
| PUMA | arithmetic secret sharing | s0-s1 | 2.3x host | 8.23 | 3 |
| BumbleBee | HE and MPC | b0-b8 | 10.6x host | 3.55 | 3 |
| BOLT | HE and MPC | s0-s2 | 6.3x host | 126.8 | 3 |
| MOAI-CPU | FHE | m0-m26 | 7.5x host | 2.14 | 1 |
| ARION | FHE | a0-a8 | 5.0x host | 13.01 | 1 |
| SHAFT-GPU | arithmetic secret sharing | s0-s3 | 18.4x VRAM, 2.0x host | 1.74 / 5.80 | 3 |
| SIGMA-GPU | FSS | s0-s6 | 65.0x VRAM, 16.1x host | 2.92 / 9.17 | 1 |
| MOAI-GPU | FHE | s0-s13 | 2.0x VRAM, 1.6x host | 2.74 / 109.03 | 1 |
| SHAFT-CPU-ViT | transfer of SHAFT-CPU | v0-v1 | 2.8x host | 13.06 | 3 |
| SHAFT-GPU-ViT | transfer of SHAFT-GPU | vit_v0-vit_v1 | 3.1x VRAM, 1.8x host | 7.89 / 15.81 | 3 |

Why each line ends where it does is in the system's `off-path.md`, together with every measured
change that is not a step of the line, such as trades of accuracy or security and runs kept as
evidence for a decision.

## The reduction procedure

Every published line follows this rule, and a step that does not is not published as one. The
objective is to minimise the pair `(VRAM peak, host peak)`. Each component is a maximum over its
own timeline, the two are never summed, and a single-pool CPU system has no VRAM component.

### The rule

From the unmodified system, repeat the following.

1. Measure both pools in one run.
2. Decompose each pool at its own peak instant.
3. Choose the pool with the larger `W = (peak - library overhead) / target`. The library overhead
   is the loaded code of the process. The choice is a reporting convention and does not force a
   sequence.
4. Within that pool, rank the objects by size at the peak and take the largest with a free fix, a
   fix that costs no measurable run time.
5. Re-measure and repeat. A pool at its target leaves the comparison. Where the peak is a plateau of
   phases within the spread, a step may lower its own phase while leaving the global peak inside
   the spread; it declares the phase (`# phase_effect: <marker>`) and is published as a plateau step.
6. When no free fix remains, take paid fixes, ranked by memory saved per unit of run time, until
   the targets are met or no admissible step remains. A line may end above its target when the
   excess above the target at the peak is library scratch and allocator retention, or when no step
   lowers the peak by more than the spread. The target is taken at the granularity of the streaming
   partition the implementation processes at once, and where the line implements that partition,
   at the line's own (MOAI-CPU the bootstrap chunk of 128 of `m8`, SHAFT the 16 vocabulary slices of
   the embedding lookup, SHARK one `d`-wide truncation batch). A step that slices a target term more
   finely reduces the target's own working set and is reported off-path.
7. When no output-identical step remains, take steps that change the output while performing the
   same protocol work (phase B below).
8. A trade of accuracy, security or portability is reported in `off-path.md` and is never a step.

A free step may follow a paid one in two declared cases. Either the lever was unavailable before
the paid step (`# requires: <step>`, with an earlier off-path run that did not move the peak,
`# refuted_as: <dir> <before> <after>`), or its object was not the largest attackable object before
the paid step and is at the step's own predecessor (`# not_selectable_until: <paid step>`).

A step passing over a larger object names it (`# object_skipped: <name>`) and gives the mechanism
in `off-path.md`. Every step declares its pool and its object before it is measured
(`# pool_attacked:`, `# object_attacked:`), and the build compares both with the measurement.

### Four kinds of fix

| kind | the object | the change |
|---|---|---|
| dead | retained past its last use | free it at the last use |
| duplicated | held more than once | keep one |
| reloadable | recomputable, or readable from disk | fetch it when it is read |
| not yet needed | created before its first use | create it at its first use |

A reloadable object lowers the resident peak only when it is never built whole. Spilling a finished
object leaves its pages touched, while building each part at its first read does not.

### Two phases of evidence

| phase | `evidence` | what the step shows |
|---|---|---|
| A | exact | its output matches the baseline's at the declared gate tier |
| B | equivalent | its output differs, the transcript does not grow, and a plaintext test computes the same function as the code it replaces |

A step that repartitions work can draw different protocol randomness, so its output moves while the
computed function stays the same. Every phase A step precedes every phase B step. A phase B step may
shrink the transcript only by a declared artefact (`# transcript_artefact: <bytes> <file>`). A
single-process system declares `transcript_none` in its manifest and rests phase B on the plaintext
test.

### Two arms

The open arm covers the four primary systems (SHARK, SHAFT-CPU, BumbleBee, MOAI-CPU), their GPU
lines and the dealer lines. A step there may be found by any means. The closed arm covers LLAMA,
PUMA, BOLT and ARION. A step there applies a lever already measured on a primary system or one of
its GPU lines, names its donor step and the property of the receiving system that makes it apply,
and is reported as a transfer, a no-op or a regression.

## Measurement

### Peak and decomposition

The recorder rides in the published run, so the peak and the object table describe one instant, and
its own table is subtracted from the peak. The host peak is the largest `VmHWM` over the poller's
samples, a slight lower bound because the kernel sets it from approximate per-CPU counters (drops of
up to about 43 MB). The device peak is read from the allocator of each GPU system. A decomposition
sampled more than 10% below its own peak is refused as misaligned.

Host objects are ranked in resident bytes. The native recorder (`tools/attrib_resident/`) records
address, size and call site of every allocation of at least 64 KiB and reads their resident pages
from `/proc/<pid>/pagemap`. `SNNI_PM_FULLSCAN` locates the remaining resident memory without naming
it, and those rows keep the `anon:` prefix and are never step targets. Other runtimes have their own
producers of the same record, the Go heap dump for ARION, a Python census for SHAFT, a recording
SEAL pool for MOAI-CPU, and a CUDA recorder for the GPU systems. Allocations are grouped by
allocating function, or by the first caller inside the measured program where the allocator is a
pool or a library (`SNNI_PM_DEPTH`).

The Go producer writes a heap dump at a new maximum only when that maximum lies at least a set step
above its previous dump (`SNNI_CENSUS_PERMILLE` in tenths of a percent, otherwise
`SNNI_CENSUS_PCT`), because each dump stops the program for up to about a minute and a half. The
last dump therefore lies less than one step below the peak.

### Atomic capture

The process is stopped for each capture (`SIGSTOP`, then `SIGCONT`) so the objects, mappings and
libraries describe one instant. The stopped time is recorded and subtracted from the wall-clock
time before a step's run-time change is computed. All parties of a multi-party system are stopped by
one signal. With `SNNI_PM_FREEZE_MODE=maxparty` the capture is triggered when the maximum over the
parties reaches a new high, and the table of the party holding it is taken.

### Replicates

Each step runs three times, independently launched. The published peak is the median replicate's,
so the peak and its decomposition come from one run, and the spread is published in full. A system
may lower this to one run with `replicates: 1` and a reason in its `MANIFEST.md`. On a multi-party
system a run's peak is the maximum over its parties, and `peak_party` names whose it is.

### The correctness gate

Seeds are pinned and declared in `MANIFEST.md` beside the gate tier.

| tier | meaning |
|---|---|
| T1 | byte-identical output |
| T1s | byte-identical structural observable where no value observable reproduces, admitted after a discrimination test |
| T2 | equal within a declared tolerance, relative or absolute |
| T3 | predicted label only |
| T0 | none, and declared as such |

### Classifying a step

`cost_type` compares the step's run-time change with its own wall spread. A step is `free` inside
the spread, `paid` above twice the spread and `marginal` between. A step that lowers one pool while
raising the other beyond that pool's spread is `cross_pool`.

Run times are measured with the recorder in the process. The stopped time of the captures is
subtracted, and the cost of intercepting allocations is not. A step that changes the number or the
pattern of allocations can change that cost, so a run-time change, and the `cost_type` derived
from it, describes the instrumented program.

## Seeds

Each system is run twice unchanged before its first step and the gate observables compared. The
seed values are in each `MANIFEST.md`. `tools/preflight.sh` reads the status column of this table
and passes a system only when its cell opens with `**verified`.

| system | what is pinned | status |
|---|---|---|
| llama-dealer | `LlamaConfig::prngs[]` to `0xdeadbeef...` | **verified deterministic** |
| shaft-gpu | CrypTen seeds pinned after `crypten.init()` | **verified 2026-07-29** |
| shaft-cpu | CrypTen seeds pinned after `ct.init()` | **verified 2026-07-30** |
| shark | dealer `0xdeadbeef`, online `srand(0x5EED)` | **verified 2026-07-30** |
| puma | plaintext side pinned (numpy, jax, `spu_public`); secret shares unpinnable | **verified 2026-08-02, reproducible within a mechanism-derived bound, T2 absolute** |
| bumblebee | plaintext side pinned; shares unpinnable, protocol approximate by design | **verified 2026-07-30, reproducible within a mechanism-derived bound, T2 absolute** |
| bolt | `prg128`, `prg256` and `seal_prng` pinned | **verified 2026-08-02, reproducible within a mechanism-derived bound, T2 absolute** |
| llama | `prngs` `0xdeadbeef` and `srand(0x5EED)` | **verified, value observable degenerate, structural gate (T1s) on the transcript** |
| sigma-gpu | `srand(0x5eed)`; key generation in the library | **verified 2026-08-05, output zero by construction, structural gate (T1s) on the transcript** |
| moai-gpu | seed pinned at `prng.cuh`, key positions through `SNNI_SEED` | **verified 2026-08-04 at the key, structural gate (T1s) after a discrimination test** |
| moai-cpu | SEAL PRNG factory pinned before the `SEALContext` is built | **verified across two images at one run** |
| arion | not pinned, because pinning would patch third-party cryptography | **verified 2026-08-03, reproducible within a mechanism-derived bound, T2 absolute** |

## Transfer lines and sequence length

ViT-Base/16 keeps BERT-base's layer dimensions and changes the sequence length (197) and the
embedding (a patch projection). SHAFT loads and traces the real model, so its two ViT systems
(`shaft-cpu-vit`, `shaft-gpu-vit`) measure a transfer to another architecture, baseline and
endpoint only. SHARK, BumbleBee, SIGMA-GPU and MOAI-GPU compute from shapes and load no model, so
their ViT runs change only the sequence length or the packing. These runs are off-path runs of
their systems (`off-path-runs/p_seq197/` for SHARK and BumbleBee, `p_seq256_vit/` for SIGMA-GPU,
`p_vit_geometry/` for MOAI-GPU), and each carries a `RESULT.md`.

## Bringing a new system onto the harness

The work on a new system ends when `tools/preflight.sh <system>` passes and the baseline is
measured. Steps are chosen one at a time from the measured decomposition, and no step is planned
before its predecessor is measured. The campaign-wide files are not edited per system
(`tools/palma/campaign.env`, `tools/palma/measure.sh`, `tools/palma/run_step.sh`,
`tools/make_steps.py`); a system keeps only `systems/<system>/` and `tools/palma/conf/<system>.conf`.

1. Copy a conf (`tools/palma/conf/llama-dealer.conf` for one pool, `shaft-gpu.conf` for two) and
   set `WORKDIR`, `SBATCH`, `DERIVE` and `CHANNELS`. Both recorders ride in one run.
2. Choose `SNNI_PM_DEPTH` by raising it until the top rows stop being library frames.
3. Build the image in a job and check that the recorder loads inside it
   (`tools/attrib_resident/build_in_image.sh`).
4. Fill the `TODO`s of `tools/palma/templates/system.sbatch`. Pass the environment through an inner
   shell, attach the poller to the process that holds the memory, and fail the run when `gate.txt`
   is missing.
5. Pin in `MANIFEST.md` the target with its derivation from source, the gate tier and observable,
   and the seed. Run the system twice unchanged and add its row to the seed
   table above.
6. Run `tools/preflight.sh`, the baseline, `tools/harvest_step.sh` and `tools/make_steps.py`, then
   read `named_frac`, the gap between the polled peak and `VmHWM`, and the recorder floor before
   choosing the first step.

Four signs mark a run that looks right and is not. A lever that measures exactly the baseline did
not load, so each lever prints a marker and the run fails without it. A step whose peak equals its
predecessor's ran a stale binary, which `binary_md5` shows. Attribution above 100% means the table
and the snapshot come from different instants. A run that finishes far too fast ran another test.

## Layout

```
README.md                 this file
tools/                    poller, object recorders, table builders, cluster scripts
systems/<system>/
  MANIFEST.md             machine, workload, targets, gate, build declarations
  steps.csv, STEPS.md     the line, machine-readable and readable
  off-path.md             measured changes outside the line, and where the line ends
  off-path-runs/<name>/   raw runs cited from off-path.md, each with a RESULT.md
  steps/<step_id>/
    levers.env            the step's settings and its declarations
    impl/                 the code that built the step, with BUILD.md
    logs/                 poll trace, peak snapshots, RUN_META, PEAK.txt
    peak-objects.md       the object decomposition at the step's peak
```

## Checking the published numbers

Every row of `steps.csv` and every `peak-objects.md` is generated by `tools/make_steps.py` from the
logs beside it. The following commands check that the published files regenerate and that every
step follows the procedure.

```
python3 tools/make_steps.py systems/<system> --check
python3 tools/make_impl.py systems/<system> --check
bash tools/test_conformance.sh
bash tools/check_policy.sh
```

`make_impl.py --check` regenerates each step's `impl/` from the declarations in `MANIFEST.md` and
`levers.env`. Container images are not part of the repository; each step's `BUILD.md` names the
pinned upstream commit and the build commands, and each run's `RUN_META` records the image digest.

## Hardware

All runs used the PALMA cluster of the University of Münster, one machine type per system.

| systems | machine |
|---|---|
| SHAFT-CPU, SHAFT-CPU-ViT, PUMA, BumbleBee, LLAMA, LLAMA dealer | 36-core Skylake, 95 GB (`normal` and `express` partitions) |
| SHARK, SHARK dealer | AMD EPYC 7742, 496 GB (`zen2-128C-496G`) |
| MOAI-CPU | 192-core AMD Zen 4, 740 GB (`zen4`), 16-core cpuset on a shared node |
| BOLT, ARION | 72-core Skylake, 1.5 TB (`bigsmp`) |
| SHAFT-GPU, SHAFT-GPU-ViT, SIGMA-GPU, MOAI-GPU | one full H200 (`gpuh200`) |

## License

The harness, the tools, the documentation and the measurement data are under the MIT License
([LICENSE](LICENSE)). Changes to third-party code under `systems/*/impl/` and the step `impl/`
directories remain under the license of the upstream project; [NOTICE](NOTICE) lists each upstream
and its license, including the three upstreams that carry none.
