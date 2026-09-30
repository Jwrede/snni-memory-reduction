# SHARK: off-path

Measured changes outside the line `s0`-`s5`, dispositions, and where the line ends.

## Status

| | |
|---|---|
| line | `s0_default` 58,089,352 kB -> `s5_up_split` 469,792 kB, 123.65x |
| `W` | 273.53 -> 2.19 (target 212,352 kB) |
| steps | `s1`, `s2` free; `s3`, `s4` paid; `s5` free (admitted after the paid steps, README.md) |
| gate | T1 `pass` on every step: `fnv1a64=4a298ab18dac1ac5` over 98,304 elements, both parties, all replicates |
| measured | 2026-09-30, T=16, `freeze_mode=maxparty`, n=3 |
| rank 1 at `s5`'s peak | `anon:[heap]`, 161.5 MB, 33.6% (leader table): not a target |

Geometry probe at 197 tokens: `off-path-runs/p_seq197/RESULT.md` (diagnostic).

## Trades

None. Named in advance as off-path:

| candidate | reason |
|---|---|
| narrower ring than `u64`, `f = 16` | precision trade |
| dropping or thinning MAC authentication (tags are half of each authenticated share, about half the matmul key material) | security trade |

### `batch_check()` frequency is not a security trade

- `batch_check()` verifies a random linear combination of the MAC residuals since its last call
  under a jointly committed seed (`common.cpp`, `always_assert(batchCheckAccumulated + peer.first == 0)`).
- Twelve calls instead of one verify every residual; soundness error ~`2^-64` -> ~`12 * 2^-64`;
  detection moves to the layer that caused it.
- Cost: eleven extra commit-and-open exchanges (hash plus 128-bit value). `s2_batch_check_stream`:
  -2.2% wall at 7.911% spread, free.
- Upstream's deferral is a speed choice; the step is a Pareto move. Earlier campaign: "batch_check
  frequency does not change security (only detection timing)".

## The disposition that licenses this line's ordering

### The object

| | |
|---|---|
| row at `s0`'s peak | `heap:comm.hpp:266:shark::Dealer::Dealer(std::string, bool)`, 56,416.3 MB, 0.948 of the peak |
| content | one `new char[size]` in the `oneShot` branch: the whole preprocessing file, 56,416,290,832 B on party 0 (`server.dat` byte for byte) |
| `named_frac_host` at `s0` | 0.9763 |

### The fix

- `Dealer(filename, oneShot = false)` builds a `FileBuf` instead of a `MemBuf`. Consumers read
  through `keyBuf->read(...)` strictly sequentially; nothing seeks or re-reads; `isMem()` is called
  nowhere. The slurp comes from `init::from_args` passing `true` (`init.cpp:70`).
- Measured from the T=4 baseline: 58,098,544 -> 3,005,284 kB, gate byte-identical, 3 replicates.

### Cost, three mechanisms (`off-path-runs/`, T=4)

Online phase only, freeze time subtracted, against the T=4 baseline's 83.7 s program time (one run's
wall paired with another's freeze; against the median run's 85.7 s each figure is about two points
lower). All gate clean.

| mechanism | read buffer | peak_host kB | runtime | own wall spread | cost_type |
|---|---|---:|---:|---:|---|
| `oneShot=false`, stock stream buffer | 8 KiB (libstdc++ default) | 3,005,284 | +26.9% | 3.478% | paid |
| the same, 8 MiB buffer | 8 MiB | 3,014,228 | +6.4% | 1.031% | paid |
| the same, 64 MiB buffer | 64 MiB | 3,072,456 | +8.5% | 2.020% | paid |

- 8 KiB refills cost ~6.8 million `read()` calls for 56 GB; 8 MiB removes 99.9% of them and 20.5 of
  26.9 points. 64 MiB is worse than 8 MiB (+2.1 points, +58 MB peak; the buffer is resident). The
  residual ~6.4% is the per-use kernel/filesystem crossing, not the syscall count.
- `mmap` with `MADV_DONTNEED` behind the read pointer is excluded: file-backed residency is counted
  as library overhead and subtracted from the peak (reclassification, not reduction). Earlier
  campaign: a SHARK run with keys in page cache read 2.8 GiB RSS while holding ~105 GB of key
  material.
- Published as `s3_key_stream`, the first paid step, built on `s2_batch_check_stream`.

### Clock

`wall_s` is the online phase only (~307 s dealer, then ~110 s online). On the total clock the
reload reads +1.6% (free); on the online clock +6.4% (paid). The measured role is online.

## Dispositions of larger objects

| step | object attacked | larger object at its predecessor's peak | disposition |
|---|---|---|---|
| `s1_layer_weights` | `std::vector<shark::span<unsigned long>>::operator[]`, 679.7 MB | `shark::Dealer::Dealer`, 56,416.3 MB | fix is paid (above); taken as `s3_key_stream` |
| `s2_batch_check_stream` | `_Vector_base<unsigned __int128>::_M_allocate`, 518.2 MB at `s0` | `shark::Dealer::Dealer`, 56,416.3 MB | same |
| `s3_key_stream` | `shark::Dealer::Dealer`, 56,416.3 MB | none | free fixes exhausted at `s2`'s peak: every other nameable object is at most 75.5 MB and smaller than the 962.9 MB of `anon:[heap]` |

- `s1` and `s2` declare `# object_skipped: shark::Dealer::Dealer` in `levers.env`.
- `anon:[heap]` (832.5 MB at `s0`) is never a target.
- `s1`/`s2` order follows the 16-thread baseline table (weights 679.7 MB above the MAC-tag buffer
  518.2 MB). The earlier order: `off-path/s1_batch_check_stream_before_weights/`.

## Rule change this system forced

`make_steps.py` first required a step to attack the largest object at its predecessor's peak.
README.md says "the largest with a free fix". SHARK's largest object (94.8%) has only a paid fix,
so no ordering passed both `NON-CONFORMANT OBJECT` and `FREE AFTER PAID`. The check now accepts a
smaller object when the larger one is declared `# object_skipped: <substring>`; the mechanism must
be in this file.

## Where this line stops

| run | change | result |
|---|---|---|
| `off-path-runs/s6_down_split` (T=16) | split the FFN down-projection (twin of `s5`) | peak 469,944 against 469,792 kB (+0.03%); published party's `anon:[heap]` 161.5 -> 296.8 MB |
| `s6_down_split_t4` (T=4) | same | `recv_array<unsigned __int128>` 91.3 -> 0 MB, Eigen scratch 49.6 -> 0 MB, `anon:[heap]` 162.1 -> 297.4 MB, peak 470,412 -> 470,688 kB |
| `t1_malloc_trim` (T=4, on s6) | six `malloc_trim(0)` calls | 56.6 of 297.4 MB returned, peak -36.5 MB, +11.0% program time; targets `anon:[heap]` |

- The freed objects stay with glibc. What remains is allocator retention, not a nameable object.
- The earlier campaign's endpoint (405 MB decimal) is 74.2 MB lower: about half trim, about half a
  QKV split of the same kind as `s6`.

## Leads from the earlier campaign (retired instrument: 1 MiB floor, no freeze)

| mechanism | finding | status here |
|---|---|---|
| thread count | T4/T8/T16 moved the peak < 0.3% at every step but one; per-thread arenas +0.03% over 4x threads | not a lever; threads pinned at 16 |
| one thread-driven win | 179 MiB at T8 on one step, non-monotonic glibc arena effect; T8 range 69 MiB against 1 MiB at T4/T16 | allocator artefact |
| `recv_array` managed-span patch | also present in the default build; an OpenMP crash fix | in the baseline, not a step |
| heap DCF traversal workspace | does not exist (stack only); the transient is `lrs::eval` spans and receive/batch-check buffers | nothing to attack |
| seed-compressing FSS keys | already seed-compressed | not attempted |
| splitting the widest truncation | refuted by that campaign's measured peak | consistent with the target derivation |

## Diagnostic: a second input (`off-path-runs/p_input2`)

`s0` and `s5` rebuilt with fill salt `0xB0B0B` instead of `0xA11CE`: peak +0.003% (`s0`) and
+0.026% (`s5`), inside the replicate spread.
