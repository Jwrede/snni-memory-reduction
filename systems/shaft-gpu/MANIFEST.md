# SHAFT-GPU: manifest

| | |
|---|---|
| paradigm | pure MPC, CrypTen as vendored by SHAFT |
| parties | 2, sharing one GPU |
| model | `andeskyl/bert-base-cased-sst2`, 12 layers, sequence length 128, batch 1 |
| encrypted | model and inputs, on CUDA |
| pools | VRAM and host |
| related | SHAFT-CPU runs the CPU role of the same codebase |

## Build

```
impl_upstream: https://github.com/andeskyl/SHAFT.git
impl_commit: 3ceb0eea7437928be4b852ed4133cabb580a52fb
impl_build_docker: docker build -f impl/Dockerfile.gpu .
impl_build_palma: sbatch --export=ALL,STEP=<step_id> impl/build_sif.sbatch
```

- Commit source: `git rev-parse HEAD` in `/root/shaft` of the measured image (see Image), read
  2026-10-01; the image was built from a shallow clone of the default branch. Same commit as
  SHAFT-CPU. `impl/Dockerfile.gpu` checks it out.
- Each step's `RUN_META` names the image (`SIF=`).
- A step is the driver plus the lever modules named in `impl/stepenv/<step>.env` (snapshot
  declaration).

## Hardware

| | |
|---|---|
| cluster | PALMA, partition `gpuh200` |
| device | one whole NVIDIA H200, 141 GB, `--gres=gpu:1` |
| host | 8 cores, 32 GB requested, on a 128-core / 1547 GB node |
| threads | `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS` follows; pinned in `impl/run_gpu.sh` (value of the upstream run script) |

- Both parties share the card. VRAM is read from each process's torch caching-allocator counters
  (per process); `nvidia-smi` is not used.
- Until 2026-08-01 the system ran on a `gpuh200mini` MIG slice. The same `s1_limb_loop` lever
  measured +74.8% and -40.3% wall on the same day there (VRAM byte-identical). The whole line was
  re-measured on the full card; MIG runs are not retained.

## Target

```
target_vram_kb: 217967
target_host_kb: 217967
overhead_vram_kb: 0
overhead_host_kb: measured_per_run
replicates: 3
gate_tier: T2
gate_tolerance: 1e-6
seed: crypten.manual_seed(0xDEADBEEF+rank, 0xC0FFEE+rank, 0x5EED) + torch.manual_seed(0x5EED)
gate_observable: the decrypted output logits, printed as `GATE|logits|...` by the baseline script
```

Instant: the embedding lookup, at the partition `s3_embed_chunk` computes it in: 16 slices of 1813
vocabulary rows, each with its own Beaver triple, table share and one-hot matrix whole
(`embed_chunk_gpu.py`). 8 B per 64-bit share. Same target as SHAFT-CPU (derivation and layer phase,
about 79.4 MB: `systems/shaft-cpu/MANIFEST.md`). The term may sit in either pool, so both pools
carry it.

```
table share                               = 28996 * 768 * 8  = 178151424 B
one-hot matrix                            = 128 * 28996 * 8  =  29691904 B
one slice's triple: one-hot slice mask    = 128 * 1813 * 8   =   1856512 B
                    table slice mask      = 1813 * 768 * 8   =  11139072 B
                    product mask          = 128 * 768 * 8    =    786432 B
slice output and running sum              = 2 * 128 * 768 * 8 =   1572864 B
target                                    =                    223198208 B = 217967 kB
```

| overhead | value | reason |
|---|---|---|
| VRAM | 0 | the device peak is torch's `peak_reserved_kb`; the CUDA context (~650 MiB per process) is created outside the caching allocator and is not in it |
| host | file-backed total of the step's own peak smaps, per run | loaded libraries |

## Image

```
shaft-gpu-estream.sif
sha256 64bd408c5d063056400a03798be60555a7011f5287609f123cb09bd351f93c74
SHAFT upstream commit 3ceb0eea7437928be4b852ed4133cabb580a52fb
```

- One image for all steps; levers selected by environment variable.
- HuggingFace weights cache in the image (828 MB); runs are offline.

## Instrumentation

| pool | mechanism | cost |
|---|---|---|
| VRAM | torch `_record_memory_history(enabled="state")` plus a watchdog that snapshots at each new allocation maximum | none measurable; host peak 2783276 kB with it, 2836292 kB without |
| host, libraries | external poller at `POLL_MS`, smaps at each new RSS maximum; file-backed mappings are the loaded libraries | none |
| host, objects | `attrib_resident`: `pmrec.so` records allocations >= 64 KiB, `pmsample` counts their resident pages via `/proc/<pid>/pagemap` | its own table, subtracted by name |

- One run carries both pools and both tables (`CHANNELS="-:PMREC=1,ATTRIB=1,SNNI_COMBINED=1"`,
  since 2026-08-01). Tested in jobs 45560087/207/208: no deadlock, gate byte-identical, VRAM
  byte-identical at 6971392 kB.
- Each pool is decomposed at its own peak; the pools are never summed.
- Capture: `freeze_mode=maxparty` (README.md, Atomic capture).
- One recorder table per party (`SNNI_PM_TABLE` expands `%d`); the poller attaches to the party
  pids the harness reports.
- The host peak is a transient; `VmHWM` gives its value, and the gap to the polled maximum is on
  the row.
- Recorder recursion guard uses `initial-exec` TLS (plain `__thread` allocates its block through
  malloc on first access; fatal under `import torch`). Verified in the image.
- Python processes: torch tensors resolve to `c10::alloc_cpu`; CPython's own allocations collapse
  onto `_PyObject_Malloc`.
- The device snapshotter runs in every step including the baseline, so its cost cancels in
  `runtime_delta_pct`.

## Reproducing a step

```
ssh palma '<camp>/harness/run_step.sh shaft-gpu <step_id>'      # 3 replicates, serial chain
sbatch --export=ALL,STEP=<id>,RUN=<n> derive.sbatch            # both pools, both tables; chained by the driver
tools/harvest_step.sh systems/shaft-gpu <id> palma <remote-log-root>
python3 tools/make_steps.py systems/shaft-gpu
python3 tools/make_steps.py systems/shaft-gpu --check          # must report no mismatch
```

`file:` rows (smaps) and object rows (recorder) come from the same run.

## Correctness gate

| | |
|---|---|
| tier | T2, relative tolerance 1e-6 |
| observable | decrypted output logits; the `GATE|logits|` line was added to the baseline script, which printed only the label |
| tolerance above 0 | fixed-point arithmetic over shares; `s1_limb_loop` accumulates limb products sequentially where the stock path batches them |
| phase A, `s1` | gated against the baseline |
| phase B, `s2`, `s3` | same exchanged bytes plus `impl/equiv_per_layer_gpu.py` (`s2`, with a declared transcript artefact) and `impl/equiv_embed_chunk.py` (`s3`) |

## Replicates

- 3 per step. The device pool reproduces to the byte across runs and machines; the host pool does
  not.
- Per-step values: `steps.csv`, `STEPS.md`. End of the line: `off-path.md`, Status.
