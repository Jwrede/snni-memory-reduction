# m1_gelu_output_release: release fired, peak unchanged

Job 46426206, 2026-08-26/27, `zen4` exclusive (r19n25), two layers, T=16, rc=0, wall 38,773 s.
Withdrawn before publication. `levers.env` beside this file, unedited.

```
m0_default                402,401,204 kB   wall 38,747 s
m1_gelu_output_release    403,454,212 kB   wall 38,773 s

peak  +1,053,008 kB  = +0.2617%
wall        +26 s    = +0.0671%
```

- `LEVER|gelu_output_release|ciphertexts=3072`; gate bit-identical over 7,680 values.
- Falsification criterion in `levers.env`: "REFUTED IF the peak moves less than this system's own
  run-to-run spread" (0.53 pp). Refuted.

Layer 0 against `m0`:

```
after_layernorm1   326,622,696 -> 328,092,928   +0.450%
after_gelu         367,744,132 -> 368,911,988   +0.318%
after_final        369,327,260 -> 370,511,500   +0.321%    <- past the release
after_bootstrap3   384,880,656 -> 386,772,256   +0.491%
after_layernorm2   401,623,372 -> 403,222,768   +0.398%
```

## Releases on this line at the time

```
p_pool                  2026-08-06  from source: the program does not use SEAL's MemoryManager,
                                    so it runs in the global profile, which keeps every block
m1_rtn_release_early    2026-08-18    768 bootstrap outputs released, -3.88%  (bigsmp, retired
                                    generation)
m2_x_copy_release       2026-08-24    768 ciphertexts released, bit-identical, -0.12%  (zen4)
m1_gelu_output_release  2026-08-27  3,072 ciphertexts released, bit-identical, +0.26%  (zen4)
```

The one that moved releases memory held in the peak-forming phase (bootstrap outputs; the peak is at
`after_bootstrap4`); the others release memory already dead by then.

## Pool samples

`impl/patch_poller_pooltrace.py` keeps every pool sample (708 live tables). Two, 68 s apart:

```
before   rss 370,445,964   203,741.1 MB  10,415 items  gelu_others.hpp:143:gelu_v2
after    rss 370,541,928   205,972.2 MB   7,527 items  test_full_scheme.hpp:519
                                                         <- create_galois_keys
```

- Rank-1 name and count change while the mass barely moves (207.3 -> 210.2 GB). This run's peak table
  ranks `test_full_scheme.hpp:519` (218,821.6 MB); `m0`'s ranks `gelu_v2` (234,877.6 MB).
- Three explanations (stale attribution, marker order, double counting) were refuted on 2026-08-25.
  Settled later as the peak plateau (`../../off-path.md`, Instrument history).
