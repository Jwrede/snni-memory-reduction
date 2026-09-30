# SHARK: off-path runs

Same procedure and gates as the line; `make_steps.py` reads only `steps/`.

| run | content |
|---|---|
| `s1_key_stream`, `s1b_stream_buffered`, `s1c_stream_buf64` | key streaming from the baseline, three read buffers (T=4) |
| `s6_down_split` | FFN down-projection split on `s5`, T=16 (`RESULT.md`) |
| `s6_down_split_t4`, `t1_malloc_trim` | the same at T=4, and with `malloc_trim` |
| `s0_norecorder` | baseline without the recorder |
| `p_input2` | second input at `s0` and `s5` |
| `p_seq197` | sequence length 197 at `s0` and `s5` |

## Key streaming, three mechanisms (T=4)

Object: `heap:comm.hpp:266:shark::Dealer::Dealer(std::string, bool)`, 94.8% of the baseline peak, one
`new char[size]` holding the whole preprocessing file. Fix in all three: `oneShot = false`
(`FileBuf` instead of `MemBuf`); only the stream's read buffer differs.

| run | read buffer | peak_host kB | runtime vs baseline | own wall spread | cost_type | gate |
|---|---|---:|---:|---:|---|---|
| `s1_key_stream` | 8 KiB (libstdc++ default) | 3,005,284 | +26.9% | 3.478% | paid | pass |
| `s1b_stream_buffered` | 8 MiB | 3,014,228 | +6.4% | 1.031% | paid | pass |
| `s1c_stream_buf64` | 64 MiB | 3,072,456 | +8.5% | 2.020% | paid | pass |

- Runtime of the online parties, freeze subtracted, against 83.7 s (one run's wall paired with
  another's freeze; against the median run's 85.7 s about two points lower).
- Gate byte-identical (`fnv1a64=4a298ab18dac1ac5`, both parties, every replicate).
- 8 KiB: ~6.8 million `read()` calls; 8 MiB removes 99.9% and 20.5 of 26.9 points; 64 MiB is worse
  (+2.1 points, +58 MB). The residual cost is not syscall-bound.
- Published as `s3_key_stream` (T=16: +5.1% at 1.091% spread, paid). Disposition and the `mmap`
  exclusion: `../off-path.md`.

## `s6_down_split_t4` and `t1_malloc_trim` (T=4)

Gate byte-identical to the baseline (`62eaa4732596`), 3 replicates each.

- `s5_up_split` split the FFN up-projection; its declared object `recv_array<unsigned __int128>`
  stayed at 91.3 MB. Markers put the peak between `ffn|after_gelu` and `after_down_linear`; the
  down-projection triple is `(128*3072 + 3072*768 + 128*768) x 32 B = 91.2 MB`.
- `s6` splits the down projection over its output columns, 4 chunks of 192:

| at the peak | s5 | s6 |
|---|---:|---:|
| `heap:...Dealer::recv_array<unsigned __int128>` | 91.3 MB | 0 |
| `heap:Eigen::internal::handmade_aligned_malloc` | 49.6 MB | 0 |
| `anon:[heap]` | 162.1 MB | 297.4 MB |
| `named_frac_host` | 0.66 | 0.38 |
| peak | 470,412 kB | 470,688 kB |

`+644 kB` against a 0.168% spread: not a step.

`t1_malloc_trim`: `s6` plus six `malloc_trim(0)` calls (the earlier campaign used nine). Targets
`anon:[heap]`, so not a candidate step.

| | peak | `anon:[heap]` | program time |
|---|---:|---:|---:|
| s5 | 470,412 kB | 162.1 MB | 87.1 s |
| s6 | 470,688 kB | 297.4 MB | 88.2 s |
| s6 + trim | 434,208 kB | 240.8 MB | 97.9 s |

- Trim returns 56.6 of 297.4 MB, peak -36.5 MB, +11.0% program time; the rest is fragmentation.
- Earlier campaign's endpoint 386 MiB = 405 MB against 467,028 kB = 478.2 MB here at T=4: about half
  trim (nine calls per layer), about half a QKV split (triple 69.2 MB), which by `s6` would free
  objects without lowering the peak.

## `s0_norecorder`: recorder cost on the runtime axis

Jobs 45513192-94, 2026-07-30, baseline image `shark-s0_default.sif`, `PMREC` unset (no `LD_PRELOAD`).
Gate `62eaa4732596` in all three, byte-identical to the baseline.

| | wall | frozen | program time |
|---|---:|---:|---:|
| with recorder (published s0 at the time) | 110 / 108 / 111 s | 24.3 / 25.0 / 24.6 s | 85.7 / 83.0 / 86.4 s, median 85.7 |
| without recorder | 95 / 87 / 88 s | 22.4 / 21.2 / 21.2 s | 72.6 / 65.8 / 66.8 s, median 66.8 |

- Recorder cost 18.9 s, +28.3% of program time; spreads 3.9% and 10.1%.
- LLAMA dealer: 0.8% to 4.7% of wall (README.md). Likely mechanism, not measured: the interceptor
  runs on every `malloc`/`new`; SHARK's key material arrives as millions of sub-floor allocations
  (`MANIFEST.md`, Recorder floor). A run at a raised floor would separate interception from insertion.
- Every step carries the recorder, so the term largely cancels in `runtime_delta_pct`; not exactly,
  since a lever can change the allocation count.
- Runtimes here are not comparable with figures measured without the recorder, such as the earlier
  campaign's 55 s online time (also other CPU: EPYC 7313P Zen3, and a marker-to-marker span).
