# LLAMA online: manifest

| | |
|---|---|
| paradigm | FSS (EzPC / sytorch), online evaluation role: parties 2 and 3 (server, client) |
| model | BERT-base, 12 layers, seq 128, bw 51, scale 12 |
| related | `llama-dealer`: same implementation, other role (separate targets, peaks, lines) |
| pools | host |
| arm | closed: levers already measured on a primary system; each borrow names its donor step, mechanism and why it applies |

## Build

```
impl_base_image: the LLAMA dealer line's base image; see systems/llama-dealer/MANIFEST.md
impl_build_palma: sbatch --export=ALL,OVERLAY=<dir> impl/build_online.sbatch
```

- Baseline: the dealer line's image plus `impl/overlay/` (online path runnable and gateable); diff
  from pristine in `impl/patches/`.
- The three levers are source changes whose per-step overlays are not in the repository. Each step
  declares `# impl_lever: unpublished` and describes the change in its `levers.env`
  (`impl/levers/o1_key_stream.env` and siblings, named after the first ordering, carry reasoning only).

## Hardware

| | |
|---|---|
| machine | PALMA `express`, `--exclusive`: 36-core Skylake AVX-512, 95 GB |
| threads | `OMP_NUM_THREADS=16`, pinned (as the dealer line) |
| allocator | `MALLOC_ARENA_MAX=2` |
| key scratch | `/scratch/tmp`, ~47 GB per run, deleted after the run |

## What is measured

| phase | content | measured |
|---|---|---|
| 1 | dealer writes `server.dat` (23,003,004,064 B) and `client.dat` (23,909,760,144 B) | no (`llama-dealer`) |
| 2 | server and client run, each reading its key file | yes |

`LlamaBase::init` gives the dealer two file-backed peers and each online party a `Dealer` reading
from a file. The launcher writes party pids only when phase 2 begins.

## Target

```
target_host_kb: 26112
replicates: 3
gate_tier: T1s
seed: LlamaConfig::prngs[] pinned to 0xdeadbeef...; srand pinned to 0x5EED (see patches/)
gate_observable: the protocol traffic between the online parties, hashed in SocketBuf::read and printed as `GATE|wire|`
overhead_host_kb: measured_per_run
```

Heaviest MatMul: FFN up-projection (s1=128, s2=768, s3=3072). `GroupElement` is `uint64_t` (8 B,
`group_element.h:53`). One online party holds only its own key share:

| term | |
|---:|---:|
| one party's matmul key, a+b+c = 2,850,816 elements | 22.8 MB |
| FFN intermediate activation, 128 x 3072 | 3.1 MB |
| matmul input, 128 x 768 | 0.8 MB |
| floor | ~26.7 MB (26112 kB) |

The dealer's floor (72 MB) is about four times larger: `KeyGenMatMul` (`conv.cpp:49`) holds both
parties' keys (`k0.{a,b,c}`, `k1.{a,b,c}`) plus a temporary.

## Source of the baseline peak

`bertbenchmark.cpp` calls `llama->init(ip, true)`; `memBuf = true` reads the whole key file
(`_kBuf = readFile("server.dat")`). Same pattern as SHARK (preprocessing bulk, 95% of its peak) and
SIGMA-GPU (40 GiB VRAM pool, 20 GiB host key buffer).

## Gate: T1s (structural)

| | |
|---|---|
| value observable | 98,304 exact zeros, forced: the dealer's mask is one bit wide, values sit at and above bit 51, `outputA` reduces modulo `bitlength = 51` before subtracting (`off-path-runs/p_reveal/`); reported as `DEGENERATE GATE` |
| observable used | every byte the online parties exchange through `SocketBuf::read`, with framing (a read of n and one of m bytes must not hash like one of n+m); both parties emit a line |
| accumulators | order-dependent FNV-1a chain and order-independent sum of per-operation hashes; the reproducing one is declared |
| admission | discrimination test on the probe image at differing depth (README.md) |

### What this gate does not cover

The arithmetic on either side of the messages. FSS is data-oblivious, so a lever that changes weight
values leaves the wire byte-identical:

- The dealer line's lazy-weight lever (stub, then zeroed re-materialisation before each matmul) is
  sound for the dealer and a silent corruption here.
- Regenerating weights from the seed is also inadmissible: `initializeInferencePartyA`
  (`llama_base.h:166`) masks every layer's weights in place (`w + r`) before the forward pass
  (`off-path-runs/p_weights/`).

## Setup applied to every step (`impl/patches/online_from_pristine.diff`)

| change | reason |
|---|---|
| no `net.load("bert-tiny-weights.dat")`, `input.load("15469.dat")` | files absent from the image; bert-tiny sized at bert-base dimensions; path never exercised |
| weights not loaded | since dealer `d5` every FC layer draws its weights from its own seeded stream in `_initScale`, applied to all parties: byte-identical weights on all three parties |
| input filled deterministically | as the seed |
| `srand(time(NULL))` pinned to `0x5EED` | the key PRNG was already pinned |

Fixed randomness exists for the gate; it is not a deployment configuration.
