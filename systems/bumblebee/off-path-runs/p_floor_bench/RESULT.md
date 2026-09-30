# p_floor_bench: BumbleBee target, measured

Target micro-benchmark of the `bumblebee` target (MANIFEST "## Target", thesis Appendix B.3): one OT
instance and the HE working set of the feed-forward up-projection, built with SPU's own objects at
the pinned commit, resident footprint read from `/proc/self/status` (VmRSS) per term.

| | |
|---|---|
| source | `floor_bench.cc`, `BUILD.bazel` (target `//examples/cpp/floor_bench:floor_bench`) |
| build | `build_floor_bench.sh`: image `localhost/bb-fm64-splits-full` (`db984970f7fb`), OpenBumbleBee `47c5560`, `git stash` restores the pinned sources of `libspu/mpc/cheetah` before the build (the image carries an earlier splits experiment); `lpn_param_{10485760, ...}`, `kPolyDegree = 8192` checked in `build.log`; `bazel build -c opt`, rc 0 |
| run | `run_floor_bench.sh`: 3 x `--mode he`, 3 x `--mode he_stock`, 3 x `--mode ot` (two processes, one per party, loopback brpc); `OMP_NUM_THREADS=1` |
| machine | server C (jwmaster01), AMD EPYC (virtualized, 16 logical CPUs), 251 GB |
| method | after each term `malloc_trim(0)`; every object written, so every page is resident |

## OT term: one instance (`BasicOTProtocols(conn, YACL_Ferret)`)

One Ferret sender and one Ferret receiver adapter, each `lpn_param_.n x 16 B` = 167,772,160 B.
The buffer is allocated in the adapter's constructor but written only by the Ferret bootstrap. The
first OT requests are served from the `OneTimeSetup` reserve (470,016 entries, written into the
buffer's head); the bench requests 1,000,000 OTs per direction, which exhausts the reserve and runs
the bootstrap.

```
                       instance constructed   after bootstrap   total RSS delta (KiB)   vs 335,544,320 B
rank 0, runs 1/2/3     21,372 / 21,100 / 21,120   313,476 / 313,488 / 313,396   334,848 / 334,588 / 334,516   +2.2 / +2.1 / +2.1 %
rank 1, runs 1/2/3     20,556 / 20,652 / 20,876   313,708 / 313,620 / 313,612   334,264 / 334,272 / 334,488   +2.0 / +2.0 / +2.1 %
```

The two buffers are 327,680 KiB; the remaining ~7 MiB per party is the `OneTimeSetup` state
(base OTs, the reserve's working arrays) and the brpc and yacl state of the first exchange.

## HE term: feed-forward up-projection `(128 x 768) x (768 x 3072)`

Parameters of `CheetahDot::Impl::DecideSEALParameters(64)` and `LazyInit`: CKKS, N = 8192, coeff
modulus {59, 55, 49, 55} with special prime, 3 carried primes; a ciphertext 393,216 B, a plaintext
196,608 B. `MatMatProtocol` with that context gives sub-shape 128 x 64 x 1: 12 left polynomials,
36,864 right polynomials, 3,072 output polynomials, packed to CeilDiv(3072, 64) = 48 response
ciphertexts; CheetahDot encrypts the left operand (12 <= 36,864).

`--mode he` builds the target's terms: the encrypted activation through `MatMatProtocol::EncodeLHS`
and SEAL encryption (BumbleBee's own 12 ciphertexts), the weight matrix at dense packing
(768 x 3072 / 8192 = 288 plaintexts in NTT form at the 3 primes) and the 48 packed response
ciphertexts.

```
term                      objects   object bytes   RSS delta (KiB), runs 1/2/3
encrypted activation      12         4,718,592      7,332 / 7,440 / 7,268
weights, dense            288       56,623,104     53,100 / 53,100 / 53,100
response, packed          48        18,874,368     18,656 / 18,652 / 18,656
HE term                             80,216,064     79,088 / 79,192 / 79,024 KiB = 81.0 / 81.1 / 80.9 MB, +1.0 / +1.1 / +0.9 %
```

Every object has exactly the derived size. The per-term deltas shift between terms because SEAL's
memory pool reuses blocks freed by the encoding temporaries of the first term; the total is the
figure to read.

`--mode he_stock` holds the operation as the unmodified receiver does (`cheetah_dot.cc:352`, the
whole weight side encoded at the 128 x 64 x 1 sub-shape, the unpacked result of `Compute`):
36,864 plaintexts (7,247,757,312 B, RSS 7,081,188 KiB) and 3,072 ciphertexts (1,207,959,552 B,
RSS 1,181,380 KiB), 8.47 GB for one matmul in all three runs. This is the encoding blow-up step
`b1_dot_encode_chunk` attacks; the target excludes it.

## Target

```
OT term (per party)   334,264 .. 334,848 KiB
HE term                79,024 ..  79,192 KiB
sum                   413,288 .. 414,040 KiB = 423.2 .. 424.0 MB   against 406,016 KiB = 415.8 MB, +1.8 .. +2.0 %
```

The two terms are measured in separate processes and summed. Like the other target micro-benchmarks
the bench confirms the sizing of the terms; it does not test minimality.
