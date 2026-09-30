# SIGMA-GPU: manifest

| | |
|---|---|
| paradigm | FSS 2PC on GPU (EzPC / GPU-MPC, SIGMA); separate protocols from LLAMA |
| parties | 2, sharing one GPU; key generation and online evaluation in one run |
| model | BERT-base, 12 layers, seq 128 |
| pools | VRAM and host, each a maximum over its own timeline, never summed |

## Build

```
impl_upstream: https://github.com/mpc-msri/EzPC.git (GPU-MPC / SIGMA)
impl_commit: unpinned
impl_instrument: patches/gate_deterministic_io.diff patches/inline_01_recvbytes_tolerates_a_signal_interrupted_read.py
impl_build_palma: sbatch impl/build_step_binary.sbatch
```

- `unpinned`: the source drop is not a git repository; no upstream revision exists to cite.
  Reconstructed patches (`git apply --check` against pristine) are in `impl/patches/`.
- One pass patches one source tree cumulatively and copies a binary after each stage
  (`impl/build_step_binary.sbatch`). A step's binary carries every probe patch applied before it
  (reporting only).
- Inline heredoc levers are extracted verbatim to `impl/patches/inline_*.py` by
  `tools/extract_inline_patches.py`; `--check` compares byte for byte.
- `patch_weight_window.py` is applied after `s6` and is on no published row.
- `s7_keygen_tight` stage: applied to copies of the two files it edits, built with
  `SNNI_BUILD_ONLY=sigma-s7_keygen_tight` (job 47049672); `s0`-`s6` binaries not rebuilt (`s6` md5
  `47d93f54` unchanged).

## Hardware

| | |
|---|---|
| cluster | PALMA, partition `gpuh200`, `--gres=gpu:1` |
| device | full NVIDIA H200, both parties on one card |
| runtime | apptainer |

- The baseline allocates a fixed 40 GiB VRAM pool per party at startup (`gpu_mem.cu:50`,
  `bytes = 40*(1ULL<<30)`, `cudaMallocAsync`); a `2g.35gb` MIG slice OOMs in 10 s (job 45099383). The
  whole line runs on the full card.
- Device memory per party from each process's own allocator counters (`nvidia-smi` cannot attribute
  per party).

## Target

```
target_vram_kb: 220944
target_host_kb: 220944
overhead_vram_kb: 0
overhead_host_kb: measured_per_run
replicates: 1
gate_tier: T1s
gate_observable: the protocol's own transcript, one token per line in `logs/gate.txt` --
                 `COMM:<total bytes exchanged>` and `KEYS:<total key material>`. STRUCTURAL, and
                 admissible only after the discrimination test recorded below. The decrypted output
                 vector is kept beside it in `gate_values.txt` as evidence and is NOT the gate.
seed: `srand(0x5EED)` in the patched binary, announced as `SEEDED|srand=0x5eed|party=N` and
      verified on four unchanged runs. The stock program seeds from the clock (`sigma.cu:170`).
```

Target: GELU keys of the feed-forward activation, 226,246,736 B = 220,944 kB, the largest measured
single operation under the single-operation rule, measured with the protocol's own per-operation
harnesses (`GPU-MPC/tests/fss/*.cu`, job 45121053).

| operation | key size |
|---|---:|
| GELU (N = 128 x 3072) | 226,246,736 B |
| softmax | 168,849,640 B |
| truncate | 129,859,624 B |
| layernorm | 31,698,672 B |

- GELU is the largest single operation among those measured; the feed-forward matmul keys are not
  measured and could be larger.
- Both pools carry the target; `W_vram` is the overhead-corrected ratio: ~189.84 baseline, ~2.92
  endpoint.
- Host communication working set: instrumentation reports 393,216 B high-water, but a 1 MB buffer
  fails and 4 MB runs (4 MB is the citable value); small against the target.
- `replicates: 1`: a run takes ~30 s to ~4 min; queue slots per step bind, not machine time
  (`tools/palma/conf/sigma-gpu.conf`).

## Gate: T1s

| | |
|---|---|
| output | not usable: `GATE_NONZERO|0/98304` with the seed pinned and input and weights filled (`off-path-runs/p_det/`) |
| observable | transcript `COMM:` and `KEYS:` |
| not covered | values inside the messages (FSS is data-oblivious); excludes any lever that regenerates masked values from a seed |
| open route | repair the reveal (CUDA rebuild; diagnosis as LLAMA online `p_reveal`) |

```
COMM:1062390674
KEYS:18075947008
```

Discrimination test (`off-path-runs/p_disc/`):

| run | Total Comm | Key size |
|---|---:|---:|
| reference (seq 128) | 1,062,390,674 B | 18,075,947,008 B |
| structural change: seq 64 | 441,539,858 B | 7,978,659,840 B |
| non-structural change: 4 threads instead of 16 | 1,062,390,674 B | 18,075,947,008 B |

Setup applied to every step: `impl/patches/gate_deterministic_io.diff` (deterministic input and
weights, pinned `srand`, full output as `GATE|logits|`; `impl/patches/README.md`). Measured binary
`bin/sigma-s0_gate3`, linked as `bin/sigma-s0_default`.

## Harness

| | |
|---|---|
| object tables | from the party whose peak is published; the leader's trace directory is read from the poller's own line |
| image and binary | fall back to the baseline's artefact explicitly; refuse if missing (no silent run of the stock binary under a step's name) |
| `derive.sbatch` | reads `combined_channel=1` from `RUN_META` (a stale two-channel table once paired with a new peak, corrected 2026-08-22, jobs 46375214-46375220) |

## Instrumentation

- Campaign harness (`tools/palma/campaign.env`): recorder floor 64 KiB, full scan, freeze, 15 ms polling.
- Full scan on a CUDA process: unreadable mappings are skipped (a 512 GiB `PROT_NONE` reservation went
  from 17 ms to 620 ms per scan before that).
- `cudarec` fixes of 2026-08-09 (probe-chain reset on free; no synchronisation) do not affect this
  system's runs: `untracked_frees=0` in all eight records; device table `sites=1` to `sites=2`, peak a
  single 40 GiB allocation.

## Earlier campaign (2026-07-17 to 07-20, not published)

| | |
|---|---|
| result | VRAM 40.7 -> 1.22 GiB, host 32.9 -> 11.77 GiB; no replicates, no freeze, residual subtracted, no working gate |
| warmup | removing the 40 GiB warmup alone: VRAM 40.7 -> 1.37 GiB at no measurable run time |
| retracted | an 11.0 GiB host figure from a binary whose patch source was lost with a container; a reimplementation from pristine source measured 21.68 GiB |
