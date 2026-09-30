# LLAMA dealer: manifest

| | |
|---|---|
| paradigm | FSS (EzPC / sytorch), offline key-generation role, measured alone as party 1 |
| model | BERT-base, 12 layers, seq 128, bw 51, scale 12 |
| output | server key material 23.0 GB, client key material 23.9 GB, streamed to disk |
| related | `llama`: same implementation, online role |
| pools | host |

## Build

```
impl_base_image: built by impl/Dockerfile.base with impl/base_patch/ applied
impl_build_docker: bash impl/build_step.sh <step_id>      # needs steps/<step_id>/impl as context
```

- `impl/build_step.sh <STEP>` builds from `steps/<STEP>/impl` as Docker context.
- The per-step build contexts (Dockerfile, overlay) were made on server C and are not in the
  repository; only `d7_weight_init_free` has its `overlay.patch`. In the repository:
  `impl/Dockerfile.base`, `impl/base_patch/`, build scripts.
- `impl_fingerprint` is a source hash (`src_diff_md5`), so a reconstruction can be checked. Steps
  declare `# impl_lever: unpublished` (`d7`: `# impl_snapshot: overlay.patch`).
- Base image: `Dockerfile.debuginfo` -> `localhost/llama-00-default-g`, the stock flags plus `-g`
  (no code-generation change; debug sections are not `PT_LOAD`). After the rebuild `addr2line`
  resolves `main` to `bertbenchmark.cpp:198`; the stock image had no debug sections.
- Each step's image is `FROM llama-dealer-base` with its overlay; nothing is compiled during a
  measurement run.
- Overlays are not cumulative: `d1_select_free` takes `api.cpp` from the earlier campaign's
  `d2_selectfree` overlay but not its `module.h` (a second lever).

## Hardware

| | |
|---|---|
| machine | PALMA `express`, `--exclusive`: 36-core Skylake AVX-512, 95 GB |
| runtime | apptainer, from the images built under podman on server C |
| threads | `OMP_NUM_THREADS=16` (server C's core count) |
| allocator | `MALLOC_ARENA_MAX=2` on clean runs |
| key scratch | `/scratch/tmp`, deleted after the md5 gate each step |

`express` and `normal` nodes checked identical (r04n06, r01n10: Sockets=2, CoresPerSocket=18,
ActiveFeatures=avx512,skylake, RealMemory=95000).

## Target

```
target_host_kb: 70312
replicates: 3
gate_tier: T1
seed: LlamaConfig::prngs[] pinned to 0xdeadbeef...; prngStr(time) present but unused
seed_note: OMP_NUM_THREADS=16 is part of the seed here, see below
gate_observable: server.dat + client.dat md5 (the dealer's entire output)
overhead_host_kb: measured_per_run
```

`GroupElement = uint64_t` (8 B, `group_element.h:53`). `KeyGenMatMul` (`conv.cpp:49`) holds both
parties' keys (`k0.{a,b,c}`, `k1.{a,b,c}`) plus a temporary `c`. Heaviest MatMul: FFN up-projection
(s1=128, s2=768, s3=3072), a+b+c = 2,850,816 elements per key:

| term | |
|---|---:|
| k0 + k1 | 5,701,632 x 8 B = 45.6 MB |
| temp c | 3.1 MB |
| co-resident A/B/C_mask | 22.8 MB |
| floor | ~72 MB |

The terms sum to 71.5 MB; `70312 kB` is the rounded 72 MB (72 x 10^6 B). Library overhead: file-backed
total of each run's peak smaps.

## Thread count is part of the seed

`random_ge` draws from `LlamaConfig::prngs[omp_get_thread_num()]`. `OMP_NUM_THREADS` 16, 8, 4, 1: peak
constant within 0.3%, key material different each time (`d7_wrap_chunk` failed the same way). Thread
count is never a lever; any repartitioning of parallel work is gate-checked.

## Correctness gate

T1: key files byte-identical across all steps (`keys.md5` per step):

```
server.dat  23,003,004,064 B
client.dat  23,909,760,144 B
```

## Instrumentation

| pool | mechanism | cost |
|---|---|---|
| host, peak value | `VmHWM`, the kernel's own high-water mark | none |
| host, peak instant | `poll_peak.sh` at 100 ms; smaps rewritten at every new maximum | none |
| host, objects | `attrib_resident`: `pmrec.so` records allocations >= 64 KiB, `pmsample` counts their resident pages via `/proc/<pid>/pagemap` | its own 3076 kB table, subtracted by name |

- One run per step; recorder in the run; its table read into `peaks/INSTRUMENT_KB` and subtracted.
- Measured over six steps: runs with the recorder 2588 to 3420 kB above runs without (table 3076 kB);
  two clean runs of one step 0.03% apart.
- The peak is a monotonic ramp (key state allocated per layer, never freed): `VmHWM` equals the polled
  maximum.
- Binary identical across runs; only `LD_PRELOAD` differs. The recorder is built against the
  container's libc; its recursion guard uses `initial-exec` TLS.

## Reproducing a step

```
# on the measuring host; both scripts travel with this system
systems/llama-dealer/impl/build_step.sh   <STEP>   # image, from steps/<STEP>/impl/Dockerfile
systems/llama-dealer/impl/run_step.sh     <STEP>   # ONE run: peak and objects together
```

```
tools/derive_host_objects.sh <results>/<STEP> <results>/<STEP> \
        localhost/llama-dealer-<STEP> systems/llama-dealer/steps/<STEP>/objects_host.jsonl
tools/harvest_step.sh systems/llama-dealer <STEP> <remote> <remote-results-root>
python3 tools/make_steps.py systems/llama-dealer
python3 tools/make_steps.py systems/llama-dealer --check
```

## Baseline reproducibility before the published runs

| | peak_host_kb | |
|---|---:|---|
| 2026-07-23, earlier campaign | 4636868 | in-container rebuild, stock image |
| 2026-07-27, this campaign | 4636908 | in-container rebuild, `-g` base |
| 2026-07-27, this campaign | 4637320 | per-step image, `-g` base |

Spread 0.01%. Published baseline (PALMA, 2026-07-29): `steps.csv`.
