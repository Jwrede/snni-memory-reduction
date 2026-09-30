# SHARK: manifest

| | |
|---|---|
| paradigm | FSS with authenticated (SPDZ-style) secret sharing; C++ stack (`shark/protocols`, Eigen + OpenMP, no CUDA) |
| model | BERT-base, 12 layers, seq 128 |
| ring | `u64`, fixed point `f = 16` |
| measured role | online evaluation, parties 0 (server) and 1 (client); dealer (party 2) not measured |
| pools | host |
| upstream | `github.com/kanav99/shark` at `6957e5ed7d279d2cd9fac0b4c1a6f30a77f91bc1` |
| benchmark | `impl/src/bert_instrumented.cpp`, from `operator-bench/shark/00_default/` |
| arm | open (primary system); donor for LLAMA online |

## Build

```
impl_upstream: https://github.com/kanav99/shark.git
impl_commit: 6957e5ed7d279d2cd9fac0b4c1a6f30a77f91bc1
impl_instrument: patches/socket_full_transfer.diff
impl_build_docker: docker build -f impl/Dockerfile.step --build-arg SHARK_SRC=<file> --build-arg EXTRA_PATCH=<diff> .
impl_build_palma: sbatch --export=ALL,STEP=<step_id> impl/build_sif.sbatch
```

- Each step is one complete benchmark source (`Dockerfile.step --build-arg SHARK_SRC=<file>`)
  containing the levers of all earlier steps; lever tokens in `impl/src/` grow monotonically from
  475 lines (baseline) to 637 (`s5`).
- The `SHARK_SRC` build argument was not recorded in `RUN_META` or the image, but each step is
  still pinned two ways: its `binary_md5` is distinct and recorded per run
  (`logs/run*/binary_md5.txt`, six different values), and its source file is named per step and
  confirmed by content (the cumulative levers appear in the file and as `LEVER|` markers in the run,
  and as the `after_up_chunk0` marker for `s5`). No two steps share a binary.
- Naming: the first two line steps were reordered after their sources were written, so the published
  `s2_batch_check_stream` is built from `bert_instrumented_s2_layer_weights.cpp` and the image
  `shark-s2_layer_weights.sif`, which keep the earlier order's names
  (`steps/s2_batch_check_stream/levers.env`); the file's content is the layer-weights lever plus the
  per-layer `batch_check`, and the run emits both `LEVER|s2_layer_weights` and
  `LEVER|s1_batch_check_stream`.
- Not on this line: `bert_instrumented_s1_key_stream.cpp` (key streaming is `s3`),
  `bert_instrumented_s6_down_split.cpp` (off-path).

## Hardware

| | |
|---|---|
| machine | PALMA `zen2-128C-496G`, `--exclusive`: AMD EPYC 7742, 128 cores, 496 GB |
| threads | `OMP_NUM_THREADS=16` (campaign count of 2026-08-03; SHARK has no own setting) |
| allocator | glibc default, not pinned (baseline is the unmodified system) |
| key scratch | `/scratch/tmp`, ~100 GB per run, deleted after the run |
| binary | `-O3 -g -march=x86-64-v3`; one binary per step, copied to every node; `binary_md5` is constant across the three runs and the nodes of a step and differs across steps (the six per-step values are in each step's `logs/run*/binary_md5.txt`) |

- The baseline holds the whole preprocessing file on both online parties (earlier campaign: 54.3
  and 53.4 GB), ~108 GB at once; 95 GB nodes cannot hold it.
- `long` (192 GB) was backfill-scheduled 6.5 h out; `zen2-128C-496G` had idle nodes. Peaks are not
  comparable with the archived Intel server-C numbers.
- Published line measured 2026-09-30 at T=16, `freeze_mode=maxparty`. Earlier T=4 runs:
  `/mnt/HC_Volume_105187419/offload/superseded_leader_only/shark/`; August T=16 study:
  `steps.operating-point-t16/`. Peaks are thread-invariant within 0.4%; cost types are not.

## What is measured

| phase | parties | measured |
|---|---|---|
| 1 | dealer (party 2) writes `server.dat`, `client.dat` (`init::gen`, file-backed `Peer`s, `init.cpp:15`) | no |
| 2 | parties 0 and 1, each reading its file (`init::eval`, `Dealer` from file, `init.cpp:38`) | yes |

- The launcher writes `party0.pid`, `party1.pid` only when phase 2 begins.
- Dealer peak (earlier campaign): 199 MB against a ~177 MB dealer floor; see `systems/shark-dealer`.

## Target

```
target_host_kb: 212352
replicates: 3
gate_tier: T1
seed: srand(0x5EED) in the measured binary before init; the dealer's own PRNG is pinned in the
      library to 0xdeadbeef (init::gen, init.cpp:20)
gate_observable: the reconstructed output vector of both online parties, emitted as
      `GATE|out|party=N|n=98304|fnv1a64=...` plus the first eight values verbatim, with the full
      98,304-element vector archived beside it as output_party<N>.txt
overhead_host_kb: measured_per_run
```

Every fixed-point product is rescaled by `ars::call` -> `lrs::call`
(`include/shark/protocols/ars.hpp`), which consumes key material per output element
(`lrs::gen`/`lrs::eval`, `src/protocols/lrs.cpp`). A DCF key over `n` bits serialises
(`send_dcfbit`, `src/protocols/common.cpp`) as `n + 1` 16-byte seed blocks, `n` correction bits,
`n` 64-bit tag corrections, one output bit and one 64-bit tag, `25 (n + 1)` bytes. Bit width 64,
shift `f = 16`:

```
DCF key over 64 bits                   (64 + 1) * 16 + 64 * 1 + 64 * 8 + 1 + 8 = 1625 B
DCF key over f = 16 bits               (16 + 1) * 16 + 16 * 1 + 16 * 8 + 1 + 8 =  425 B
two authenticated Boolean shares       2 * (1 + 8)     (send_authenticated_bshare) =   18 B
four authenticated arithmetic shares   4 * (16 + 16)   (send_authenticated_ashare) =  128 B
per element                                                                     = 2196 B
```

Granularity: one `d`-wide linear output, `s * d = 128 * 768 = 98,304` elements (the partition
`s4`/`s5` compute in):

```
key material            98,304 * 2196    = 215,875,584 B
input + output buffers  2 * 98,304 * 8   =   1,572,864 B
target                                   = 217,448,448 B = 212,352 kB = 217.4 MB
```

| related quantity | value |
|---|---|
| widest truncation of the unmodified program (FFN up-projection, `128 * 3072 = 393,216` elements) | ~864 MB key material |
| measured validation (`off-path-runs/p_floor_bench/`, 3 runs) | key bytes 215,875,584 B per party = derived key term; resident 258.2 to 258.3 MB, +18.7 to 18.8% over 217.4 MB (DCF keys are larger in memory than serialized) |
| figure in `impl/patches/lrs_charge_chunk.diff`, `s4_stream_and_charge.diff` comments | ~2,299 B per element: in-memory size of the two DCF keys of the GELU's `ars::call(y, f + 3)` at 19 bits (`impl/src/bert_instrumented.cpp:166`): 64-bit key sub-arrays `65 * 16 + 64 + 64 * 8 = 1616 B`, 19-bit `20 * 16 + 19 + 19 * 8 = 491 B`, plus two 96 B `span<DCFBitKey>` entries; no Boolean or arithmetic shares |

## Source of the baseline peak

`init::eval(party, ip, port, oneShot = true)` takes the `oneShot` branch of `Dealer`
(`include/shark/utils/comm.hpp:257-281`): file size read, `new char[size]`, the whole preprocessing
file read before inference. Same pattern as LLAMA online (~23 GB key file) and SIGMA-GPU (40 GiB
VRAM pool, 20 GiB host key buffer).

## Correctness gate

| | |
|---|---|
| tier | T1, byte-identical |
| observable | both online parties emit the reconstructed output; both lines go into `gate.txt` |
| stock program | `main` computed `output::call(y)` and discarded it; the benchmark emits it |
| archive | full 98,304-element vector per party per run (`output_party<N>.txt`) |

## Setup applied to every step

| change | reason |
|---|---|
| gate observable | above |
| `srand(0x5EED)` before `init` | online parties seed from `toBlock(::rand(), ::rand())` (`init.cpp:53`) and nothing called `srand`; the run fails without the `SEEDED|` marker |
| `-g` with `-O3` | symbolisation; no change to code generation |
| `-march=x86-64-v3` instead of `-march=native` | image built elsewhere; one binary on every node |
| pinned upstream commit | the original Dockerfile cloned the default branch |
| `impl/patches/socket_full_transfer.diff` | `SocketBuf::read`/`write` asserted one `recv()`/`send()` moves all bytes (`src/utils/comm.cpp:78,84,90`); `MSG_WAITALL` returns short on a signal; job 45509359 died with the freeze poller attached. The patch loops and retries on `EINTR`; no change to buffers or bytes on the wire |
| `recv_array` managed-span fix (from `00_default`) | copies out of `MemBuf` so OpenMP threads do not race on its read position; keeps `oneShot = true`; part of the baseline |

Fixed randomness exists for the gate; it is a security defect in deployment.

## Determinism

- Dealer PRNG pinned to `0xdeadbeef` in the library; online PRNG pinned by `srand(0x5EED)`.
- `shark::span<T>(size)` is `new T[size]` (indeterminate for `u64`); weights and input are not
  written before `input::call` adds the dealer's mask. Large spans come from `mmap` (zero); the
  768- and 2304-element biases come from the `brk` heap.
- Checked by running twice unchanged: `impl/detcheck.sbatch` at one layer; authoritative check is
  the baseline's three replicates (`make_steps.py` reports disagreement).

## Recorder floor (64 KiB)

| allocation | size |
|---|---|
| `span<DCFBitKey>` | 96 B per key, one allocation (37.7 MB for 393,216 elements; named) |
| per-key sub-arrays, bit width 64 | 1040 B, 64 B, 512 B (`new[]` each; below the floor) |
| per-key sub-arrays, bit width 19 | 320 B, 19 B, 152 B |
| one wide ARS | ~167 MB above the floor, ~828 MB below |

- At an earlier baseline run (58,095,268 kB) `anon:[heap]` was 936.1 MB (1.6%),
  `object_vs_unnamed` 60.23.
- Near the target, the peak is key material allocated below the floor. The floor is campaign
  policy (README.md). The unnamed bytes can be estimated from the named `span<DCFBitKey>` count and
  the sizes above; an estimate does not make an `anon:` row a step target.
