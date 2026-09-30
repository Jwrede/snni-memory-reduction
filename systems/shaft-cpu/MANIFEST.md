# SHAFT-CPU: manifest

| | |
|---|---|
| paradigm | pure MPC, CrypTen as vendored by SHAFT |
| parties | 2, both on the host CPU |
| model | `andeskyl/bert-base-cased-sst2`, 12 layers, sequence length 128, batch 1 |
| encrypted | model and inputs |
| pools | host |
| related | SHAFT-GPU runs the GPU role of the same codebase |

## Build

```
impl_upstream: https://github.com/andeskyl/SHAFT.git
impl_commit: 3ceb0eea7437928be4b852ed4133cabb580a52fb
impl_build_docker: docker build -f impl/Dockerfile.cpu .
impl_build_palma: sbatch --export=ALL,STEP=<step_id> impl/build_sif.sbatch
```

- Commit source: `/root/shaft_commit.txt` and `git rev-parse HEAD` in `/root/shaft` of the measured
  image (see Image), read 2026-10-01. `impl/Dockerfile.cpu` checks it out.
- A step is the driver plus the lever modules named in `impl/stepenv/<step>.env`. Each step
  declares the full module set as `# impl_snapshot:`.

## Hardware

| | |
|---|---|
| cluster | PALMA, partition `express`, `--exclude=r07n04`, `--exclusive` |
| node | 36-core Skylake AVX-512 (2 x 18), 95 GB |
| threads | `OMP_NUM_THREADS=8`, pinned |

- `express` and `normal` have the same node type (`sinfo`: 36 CPUs, 2 x 18, `avx512,skylake`,
  95000 MB). `express` was used for its shorter queue.
- `r07n04` is excluded because it has 192 GB.
- 8 threads is the value of the upstream `run.sh`. Earlier campaign, this workload: 8 against 48
  threads changed wall time by 6.2 s of 224.5 s and the peak not at all. Thread count is not a lever.
- Exclusive nodes, because wall time types a step `free` or `paid`.

## Target

```
target_host_kb: 217967
overhead_host_kb: measured_per_run
replicates: 3
gate_tier: T2
gate_tolerance: 1e-6
seed: crypten.manual_seed(0xDEADBEEF+rank, 0xC0FFEE+rank, 0x5EED) + torch.manual_seed(0x5EED)
gate_observable: the decrypted output logits, printed as `GATE|logits|...` by the baseline script
```

Instant: the embedding lookup, the larger of the two phases under per-layer loading. It is taken at
the partition `s7_embed_chunk` computes it in: 16 vocabulary slices of `ceil(28996 / 16) = 1813`
rows, each slice's triple freed before the next, table share and one-hot matrix whole
(`embed_chunk_cpu.py`). One share is 8 B per value (64-bit ring). Beaver triples are produced at
use, so no cryptographic material accumulates.

```
table share                               = 28996 * 768 * 8  = 178151424 B
one-hot matrix                            = 128 * 28996 * 8  =  29691904 B
one slice's triple: one-hot slice mask    = 128 * 1813 * 8   =   1856512 B
                    table slice mask      = 1813 * 768 * 8   =  11139072 B
                    product mask          = 128 * 768 * 8    =    786432 B
slice output and running sum              = 2 * 128 * 768 * 8 =   1572864 B
target                                    =                    223198208 B = 217967 kB = 223.2 MB
```

| quantity | value | source |
|---|---|---|
| `V`, `d`, `s` | 28996, 768, 128 | `AutoConfig` of the model run |
| layer phase | 79.4 MB: weights `(4*768*768 + 2*768*3072)*8 = 56623104 B` + FFN up-projection triple `(128*768 + 768*3072 + 128*3072)*8 = 22806528 B` | smaller than the embedding phase |
| `bert-base-uncased` vocabulary (30522) | 234.8 MB, slices of 1908 rows | same expression, not the model run |
| whole encrypted parameter set | 866494480 B = 846186 kB, 201 `Parameter` modules, 3.88x the target | `impl/levers/probe_graph_owners_cpu.py` |
| library overhead | file-backed total of the step's own peak smaps, per run | excluded from `W` |
| measured validation | every term's tensor equals the derived size; resident memory +2.1 / +2.9 / +3.0% over 223198208 B in 3 runs | `off-path-runs/p_floor_bench/` |

The target follows the rule of README.md step 6: streaming a target term (table share, one-hot
matrix) more finely is off-path.

## Image

```
shaft-cpu-s0_default.sif
sha256 f35e71f6d21c683f46f1872c2cca3276aa8fdd3813e117aff97fbded42a3abfe
SHAFT upstream commit 3ceb0eea7437928be4b852ed4133cabb580a52fb
```

| | |
|---|---|
| recipe | `impl/Dockerfile.cpu`, from `operator-bench/shaft/00_default/Dockerfile`; SIF built by `impl/build_sif.sbatch` |
| changes to the recipe | numpy < 2.0 (torch 2.0.1 ABI); gcc, to build the recorder inside the image |
| model weights | in the image; runs are offline |
| steps | one image for all steps; lever modules selected by `SHAFT_LEVERS=<module>[,<module>]` |
| bind-mounted | measured script and lever modules, md5 in `RUN_META`; the launcher runs from the image |
| recorder check | in the build job: `import torch, crypten, transformers` under the preload completes, table 3072 kB |

## Instrumentation

| pool | mechanism | cost |
|---|---|---|
| host, libraries | external poller at `POLL_MS`, smaps at each new RSS maximum; file-backed mappings are the loaded libraries | wall time only, logged as `frozen_ms` and subtracted |
| host, objects | `attrib_resident`: `pmrec.so` records allocations >= `SNNI_PM_MIN`, `pmsample` counts their resident pages via `/proc/<pid>/pagemap` | its own table, subtracted by name |

- One run per replicate; the recorder rides in the measured run.
- One table per party (`SNNI_PM_TABLE` expands `%d`). The measured script writes `party<N>.pid`,
  because spawned workers' cmdlines do not contain the script name.
- Capture: `freeze_mode=maxparty` in all published runs (README.md, Atomic capture).
- `file:` rows (smaps) and object rows (recorder) come from the same run and the same party.

## Reproducing a step

```
tools/palma/deploy.sh shaft-cpu                   # sbatch + runner only
scp systems/shaft-cpu/impl/bert_instrumented_cpu.py palma:<workdir>/
scp systems/shaft-cpu/impl/levers/*.py             palma:<workdir>/levers/
scp systems/shaft-cpu/impl/stepenv/*.env           palma:<workdir>/stepenv/
tools/palma/run_step.sh shaft-cpu <step>          # 3 replicates + 3 derives, chained
tools/harvest_step.sh systems/shaft-cpu <step> palma /scratch/tmp/j_wred02/snni_campaign/shaft-cpu/results
python3 tools/make_steps.py systems/shaft-cpu
python3 tools/make_steps.py systems/shaft-cpu --check   # must report no mismatch
```

- `deploy.sh` copies only `*.sbatch` and `*.sh`; the `scp` lines copy the `.py` and `.env` files.
- The sbatch sources `stepenv/<step>.env` and refuses a step without one.

## Correctness gate

| | |
|---|---|
| tier | T2, relative tolerance 1e-6 |
| observable | decrypted output logits; the `GATE|logits|` line was added to the baseline script, which printed only the label |
| tolerance above 0 | fixed-point arithmetic over shares: reordering exact integer operations moves the last bits |
| phase A, `s1`-`s4` | gated against the baseline; a deviation fails `make_steps.py` with its magnitude |
| phase B, `s5`-`s8` | not gated against the baseline; same exchanged bytes plus `impl/equiv_per_layer.py` (`s5`, `s6`), `impl/equiv_embed_chunk.py` (`s7`), `impl/equiv_encrypt_chunk.py` (`s8`) |

Why phase B: `off-path.md`, "The constraint that shapes the lever space on this system".

## Replicates

- 3 per step, independently launched, serial, frozen.
- Published peak: median over runs of the maximum over parties, recorder table (4,100 kB)
  subtracted.
- Per-step values: `steps.csv`, `STEPS.md`. End of the line: `off-path.md`, Status.
