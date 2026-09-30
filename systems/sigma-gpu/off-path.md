# SIGMA-GPU: off-path

## Status

`s0`-`s6` re-measured 2026-09-29, `freeze_mode=maxparty`, recorder c2-32b38f744, one run per step
(`tools/palma/conf/sigma-gpu.conf`).

| step | VRAM (kB) | `W_vram` | host (kB) | `W_host` | pool attacked | cost |
|---|---:|---:|---:|---:|---|---|
| `s0_default` | 41,943,045 | 189.84 | 34,524,212 | 155.72 | | baseline |
| `s1_nowarmup` | 645,397 | 2.92 | 34,522,332 | 155.72 | VRAM | free |
| `s2_commbuf_measured` | 645,397 | 2.92 | 24,167,648 | 108.85 | host | free |
| `s3_keybuf_measured` | 645,397 | 2.92 | 22,068,544 | 99.35 | host | free |
| `s4_dealerbuf_measured` | 645,397 | 2.92 | 20,105,092 | 90.46 | host | free |
| `s5_keybuf_tight` | 645,397 | 2.92 | 19,055,892 | 85.71 | host | free |
| `s6_dealer_window` | 645,397 | 2.92 | 2,143,032 | 9.17 | host | paid (+119.7%) |

- VRAM 64.99x, `W_vram` 2.92 against the 220,944 kB target; from `s2` on `W_host` is larger, so every
  later step attacks the host pool (README.md step 3). Rank 1 at `s6`'s device peak:
  `gpu_mem.cu:68:gpuMalloc`, 100% (the pool itself).
- Host 16.11x; `s6_dealer_window` is the campaign's largest single step (-88.75%).
- No trades: every lever is sizing or placement (pool sizes, buffer ceilings, where keys live). The
  gate is structural, so "numerically inert" rests on the mechanism plus an unchanged transcript.
- Against the earlier three-run generation, peaks moved -0.07% to +0.06% (freeze leader's peaks):
  `s0` 34,522,940 -> 34,524,184 kB, `s6` 2,139,588 -> 2,132,804 kB. Device-recorder table subtracted
  since 2026-08-22 (before: ~0.26% higher). `s6`'s peak is a spike (`VmHWM` 3.1% above the poller),
  so that subtraction is an upper bound there.

## End of the line: `s7_keygen_tight`

| | |
|---|---|
| object at `s6`'s published (non-leader) peak | `SIGMAKeygen::SIGMAKeygen <- gpu_mem.cu:77:cpuMalloc`, 1,179,660 kB (job 47048628, `steps/s6_dealer_window/logs/run1/objects_host_3418849.jsonl`): the dealer's three pinned buffers, key window `sigma.h:371` 1 GiB plus two LLAMA layernorm buffers `sigma.h:392/393` 64 MiB each; `cudaHostRegister` makes them resident at full size |
| demand | largest operation 825,125,000 B (window 1 GiB); `llamaBuf` 281,856 B per round, `dummyBuf` 6,764,544 B total (64 MiB each); slack 375.8 MB |
| lever | 800 MiB, 1 MiB and 8 MiB, each with an overflow assert; `not_selectable_until: s6_dealer_window` |
| measured | 1,941,788 kB (W_host 8.25), wall -13.4% against `s6`, gate identical; dealer phase 1,807,088 kB, peak moved to the online phase |
| verdict | not a row: the peak is a short spike in key generation; the table sits 42.5% below it (limit 10%) on both measurements (`off-path-runs/s7_keygen_tight_misaligned/`) |

The leader's VmHWM (2,142,220 kB) was set in its dealer phase before its poller attached; its
table was sampled in the online phase at 57.8% of its peak, so the window was absent from it. The
non-leader climbs to 2,149,828 kB in the dealer phase and falls to 1,236,504 kB at `close()`.

## Weights: phantom rows

| row (leader's `s6` table) | size |
|---|---:|
| `heap:GPUBERT::GPUBERT <- layers.h:583:FC::FC` | 453.3 MB |
| `heap:main <- bert.h:89:GPUBERT::GPUBERT` | 226.6 MB |
| together | 679.9 MB |

- 12 blocks x (768x2304 + 768x768 + 768x3072 + 3072x768) = 84,934,656 elements x 8 B = 679,477,248 B
  (0.06% apart).
- `mincore` in the leader's online phase: zero resident pages (`Tensor2D` is `data(new T[d1 * d2])`,
  never written):

```
SNNI_MHA|wQKV |elems=1769472|bytes=14155776|resident_kb=0|mincore_rc=0|nonzero=0
SNNI_MHA|wProj|elems= 589824|bytes= 4718592|resident_kb=0|mincore_rc=0|nonzero=0
```

- Weight-window lever at three placements (`s7_weight_window`, `s8_weight_window_early`): -0.25%,
  -0.49%, null; `SNNI_WW|` reports 1.36 GB released.
- The non-leader's pagemap counts both rows as resident at full size at `s6` and `s7`. Which reading
  holds for the online-phase peak is not measured.
- `FC::FC` also allocates `weightGrad(in,out)`, `Vw(in,out)`, `Vb(out)` (`layers.h:597`): gradient
  and SGD-momentum buffers, read nowhere in GPU-MPC; address space, not resident bytes.

## How to read `W_host`

`experiments/sigma/sigma.cu`:

```
auto sigmaKeygen = new SIGMAKeygen<u64>(party, bw, scale, "", keyBufSz);
...
sigmaKeygen->close();          // writes nothing: `if (keyFile.compare("") != 0)` is false
auto sigma = new SIGMA<u64>(...);
sigma->keyBuf   = sigmaKeygen->startPtr;    // the ONLINE phase reads the same buffer
```

- Key generation and the online phase share one buffer in one process; the target assumes a dealer
  streaming keys. The remaining factor is not a count of available levers.
- Consumption is forward, single-pass: readers take the cursor by address and advance it
  (`fss/gpu_matmul.h:51` and siblings); `backend/sigma.h:135`, `:148` use one key and drop it. A window
  need hold one operation's key: `p_keywindow_high` measured 787 MiB, hence 1 GiB in `s6`.
- `close()` ends with `assert(keySize < keyBufSize)` (assert string in the binary, `NDEBUG` unset):
  an over-tight buffer aborts.

## Geometry (README.md, Transfer lines and sequence length)

| run | result |
|---|---|
| `p_seq197` | 192, 196, 197 fail (`cudaMallocAsync`, out of memory); 256 fails (`cudaMemcpy`, invalid argument) |
| `p_seq256_vit` | with `patch_keybuf_env.py` (`SIGMA_KEYBUF_MB`): `s0` 67,811,944 / 67,809,772 kB, `s6` 4,457,012 / 4,487,716 kB; 15.11x at 256 against 16.13x at 128 |

Cause: `keyBufSz = 20 * OneGB` (`experiments/sigma/sigma.cu`, bert-base branch), a model-size
constant; `p_bufhighwater` measured it 84.2% full at 128 tokens. 197 fails for a second, unidentified
reason and is padded to 256. Earlier explanations: `off-path-runs/p_seq197/RESULT.md`,
`impl/patches/patch_alloc_diag.py` header.

## Earlier campaign: key streaming from disk

| | |
|---|---|
| mechanism | online party reads its 16.83 GiB of keys with `mmap(MAP_PRIVATE)`, `madvise(SEQUENTIAL)`, an eviction thread issuing `MADV_DONTNEED` |
| result | host RSS lower; online phase ~5x slower (MHA 13.3 s against 1.2 s) |
| relation | a paid step; `s6_dealer_window` is this campaign's measured version |
| buffer optimum | keys in RAM: 4 MB communication buffer best; keys streamed: 64 MB better than 4 MB on both axes (11.77 against 12.92 GiB; 21 against 38 s) |
| open | evictor reported 16.8 GiB evacuated while ~11.7 GiB stayed resident |

- `sigma.h:91-99` (key file path) reads the whole file into `keySize` bytes: persistence, not memory.
- Communication buffer: 393,216 B high-water against 10 GiB per party (~14,000x); 1 MB fails, 4 MB
  runs, so 4 MB is the citable value; BERT-base at 128 only (larger models preset `keyBuf` at 300 to
  450 GiB).
- A first buffer sweep also set `keyBuf` to 16 GiB, below the 18,075,947,008 B = 16.83 GiB of keys
  (GB/GiB confusion); the re-run varied one thing at a time.

## Off-path runs

| run | content |
|---|---|
| `p_det` | four unchanged runs: seed and fill fixed, output zero |
| `p_disc` | gate discrimination test |
| `p_llamabuf_highwater` | demand of the two 1 GiB dealer buffers (basis of `s4`) |
| `p_seq197`, `p_seq256_vit` | geometry |
| `s6_key_window` | key file instead of the shared buffer: object -99.3%, peak -0.71% (basis of `s6_dealer_window`) |
| `s7_keygen_tight_misaligned` | above |
| `s7_weight_window`, `s8_weight_window_early` | weight-window lever |
