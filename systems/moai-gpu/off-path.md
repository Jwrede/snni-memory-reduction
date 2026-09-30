# MOAI-GPU: off-path

## Status

Ten rows, one replicate each, combined channel (peak and object list from one execution).

| step | VRAM (kB) | `W_vram` | host (kB) | `W_host` | pool attacked | cost |
|---|---:|---:|---:|---:|---|---|
| `s0_default` | 136,544,256 | 5.40 | 3,338,812 | 175.45 | | baseline |
| `s1_complexroots_free` | 135,888,896 | 5.38 | 2,105,804 | 108.55 | host | undetermined |
| `s2_boot_input_release` | 124,026,880 | 4.91 | 2,105,564 | 108.54 | VRAM | free |
| `s3_boot_layer_release` | 111,050,752 | 4.40 | 2,110,156 | 108.66 | VRAM | free |
| `s5_park_input_chunk_ffn` | 109,117,440 | 4.32 | 2,170,308 | 112.05 | VRAM | paid |
| `s6_stream_boot23` | 85,229,568 | 3.37 | 2,173,184 | 112.08 | VRAM | paid |
| `s8_stream_boot1_ln1` | 82,345,984 | 3.26 | 2,175,392 | 112.20 | VRAM | paid |
| `s9_stream_boot4` | 77,135,872 | 3.05 | 2,173,832 | 112.24 | VRAM | paid |
| `s12_ffn_threads` | 71,761,920 | 2.84 | 2,116,008 | 109.11 | VRAM | paid |
| `s13_park_encx_softmax` | 69,173,248 | 2.74 | 2,116,924 | 109.03 | VRAM | paid |

VRAM 1.974x, host 1.577x. Single run: `cost_type` declared by mechanism; `s12` (worker-count cut) is
paid by mechanism (retyped 2026-09-30).

### End of the line

| at `s13`'s device peak | MB | frac |
|---|---:|---:|
| `PhantomSecretKey::encrypt_zero_symmetric` | 42,316.3 | 0.597 |
| `PhantomPublicKey::encrypt_zero_asymmetric_internal` | 28,185.7 | 0.398 |
| `PhantomSecretKey::compute_secret_key_array` | 37.7 | 0.001 |

- The two library rows (key bank) are 99.5% of the peak; library-hard (earlier campaign's audit),
  declared `object_skipped`. Largest row beneath: 37.7 MB.
- Not built: a stream fix for pool fragmentation (below).

### Measurement properties

- The device figure is Phantom's reserved high-water; `s1` moves it -0.48% (undetermined at n=1).
  Against the earlier three-run generation, nine of ten rows agree within +-1.3%; `s9` -4.19%.
- Host: both recorders' tables subtracted (`pmtable_*` and `devtable_*`; at first only one, giving
  +1.07% to +1.33% on every row). `s13` 2,116,924 kB against a recorder-free run's 2,116,948 kB.

### Order of `s12` and `s13`

The free thread lever was first measured at the end of the line; the reservation series places its
earliest fully effective position at `s9` (from there `after_bootstrap3` is the last growth event).

```
predicted   s12 on s9    71,729,152 kB     measured 71,794,688     0.09% high
predicted   s13 on s12   69,173,248        measured 69,173,248     EXACT, to the kilobyte
```

- Both orders end on the same number (cumulative reservation conserved under re-ordering); after the
  thread step the paid step buys 2,560 MiB instead of 1,408.
- Superseded runs: `off-path-runs/superseded_order/` (old `s11` bought 1.4 GB for +2.0% wall).
- `s12` declares `requires: s5_park_input_chunk_ffn`, `refuted_as: s4_ffn_threads` (on `s3` the same
  change raised the peak 2.95%).

## ViT geometry probe (`off-path-runs/p_vit_geometry/`)

```
BERT  256 x 128   s0 136,118,272 kB   s13 69,173,248
ViT   128 x 256   s0 136,740,864      s13 69,173,248     0 kB apart
```

- Both fill 32,768 slots: the peak depends on packed volume, not on the rectangle.
- `num_row` is a source constant; the sequence length is also a literal in `softmax` and
  `softmax_boot`. A wrong-stride softmax walks the same modulus chain (T1s gate passes), so the probe
  waited for `SNNI_GEOM|slot_count=..|num_row=..|num_batch=..|num_X=..` with a hard abort on mismatch.

| system | constant tuned to 128 tokens |
|---|---|
| SIGMA-GPU | key buffer size (`n_seq` 256 runs once raised; 197 fails) |
| BumbleBee | `b8`'s single OT instance |
| MOAI-GPU | softmax batch literal, two functions |

## Host pool: blocked on identity

`steps/s2_boot_input_release/peak-objects.md`, host peak 2,104,588 kB (same rows at `s13`):

```
anon:[heap]                        1939.3 MB   90.0%   located, UNNAMED
file:nvidiactl                       75.5 MB    3.5%   library text
anon:mmap-rest(83)                   47.9 MB    2.2%   located, unnamed
Bootstrapper::genorigcoeff()         23.8 MB    1.1%   <- largest NAMED row
PhantomSecretKey::PhantomSecretKey   18.9 MB    0.9%
libcuda.so.1+0x30bdb3                11.5 MB    0.5%
```

- Largest targetable object 23.8 MB (1.1%).
- On MOAI-CPU (same recorder, same depth) `anon:[heap]` is 379.9 MB of a 393 GB peak; here 90%:
  something allocates host memory past the hooks (`malloc`, `calloc`, `realloc`, `aligned_alloc`,
  `memalign`, `posix_memalign`).
- Next action (instrument, as BOLT's `SEAL_MALLOC` -> `aligned_alloc` case):

```
nm -D <binary> | grep -E ' U (malloc|calloc|realloc|aligned_alloc|posix_memalign|memalign|valloc|_Znwm|_Znam)(@.*)?$'
```

  plus `libntl.so.44`.

## Device noise floor at one replicate

```
s2_boot_input_release   121,024 MiB
s3_inter_output_free    119,712
p_finalprobe            123,232
s3_input_copy_fused     121,824
                        -------
spread                    3,520 MiB = 2.94%
```

- `s2` and `s3_inter_output_free`: device object tables identical (110,921.1 MB, every object +0.0 MB);
  reserved high-water differs 1,312 MiB (1.08%).
- `peak_alloc_kb` 108,321,427 in all four runs (insensitive: tracks what the pool ever reserved).
- Host spread over five runs: 6,020 kB (0.29%):

```
s1_complexroots_free   2,104,176 kB
s2_boot_input_release  2,104,588
s3_inter_output_free   2,104,272
p_finalprobe           2,104,860
s3_input_copy_fused    2,110,196
```

## Pool growth events

`p_finalprobe`: the final `pool_reserved_MiB` of the marker series equals `peak_reserved_kb`; the pool
is monotone.

```
phase                 used       reserved    ratio
-> before_attention  +27,648     +53,792      1.95
-> after_layernorm1     +768      +1,408      1.83
-> after_intermediate +27,648     +4,960      0.18
-> after_gelu         +6,144      +7,616      1.24
-> after_bootstrap3  +15,360        +352      0.02
-> after_layernorm2     +768     +14,496     18.90
```

### Only reductions at or after the last growth event lower the peak

```
step                                   where it acts                 result
s2_boot_input_release   bootstrap loops, after after_gelu            -9.0%   WORKED
s3_boot_layer_release   boot_layer -> layernorm2 and bootstrap3      -10.4%  WORKED
s4_input_copy_fused     before_attention, re-ordered                  null
s5_park_input_copy      before_attention, parked after the loop       null
s6_fused_park_input     before_attention, never built at all          null
```

- `s6_fused_park_input`: -26,816 MiB at `before_attention`, later phases +26,752 MiB.
- Correction (2026-08-18, `s11_park_encx_softmax` of that ordering): on the `s9` tree the last growth
  events are `after_final` (+1,408) and `after_bootstrap3` (+1,312); `s11` removed +2,496 inside the
  attention block and kept 1,408 (56% taken back, not 100%).
- Events after `s3` at the time: `keys_created` +40,576 (permanent), `before_attention` +26,976,
  `after_intermediate` +12,448, `after_gelu` +8,544 (last event, ~-7.9% available).

### Conservation under re-ordering (`s3_input_copy_fused`, `s4_input_copy_fused`)

```
                    jump s3      jump s4     recovered
before_attention    +53,792      +43,104      -10,688   <- the lever
after_attention           0       +2,528       +2,528
after_layernorm1     +2,048       +9,440       +7,392
after_intermediate   +3,552       +4,384         +832
after_gelu           +8,416       +8,416            0   identical
                                              -------
                                              +10,752 against 10,688 withheld
```

A lever that changes only when memory is demanded leaves the cumulative reservation unchanged.
`before_attention`'s floor (27,648 + 16,128 MiB) is real; lowering it needs a different `enc_ecd_x`
level schedule.

### Reuse levers compete (`s4_inter_output_free`)

```
marker              s3        s4      jump s3   jump s4
after_gelu       108,384   100,992     +8,416    +1,216   <- -7,200 against -7,296 forecast: exact
after_bootstrap3 108,384   105,824          0    +4,832   <- was ZERO in s3
```

- Net -2,560 MiB (-2.36%), inside the 2.94% noise.
- `s3_boot_layer_release` had handed `boot_layer`'s 768 freed blocks to `after_bootstrap3` and
  `after_layernorm2`; releasing `inter_output` earlier lets GELU take them first.
- `s2_boot_input_release` released 768 MB and the pool ended 26,879.9 MB lower (a later same-shape
  allocation reused the blocks). Levers that free without handing blocks to a later request:
  `s3_inter_output_free` (null), `s3_input_copy_fused` (unresolved), MOAI-CPU `m1_boot_out_move`
  (+4.36%).

## Bootstrap working set

N = 65536: one limb = 1 MiB; a ciphertext at depth `d` is `(d+1) MiB`; arrays are 768 wide
(`level-audit.md`). At `after_bootstrap1` (pool 101,170 MiB):

```
keys                             40,325 MiB   (level-audit, 0.4% from source)
bootstrap output rtn, depth 20   16,128 MiB   = 768 x 21
bootstrap "workspace"            26,112 MiB   = 768 x 34
enc_ecd_x_copy, depth 20         16,128 MiB   = 768 x 21
att_selfoutput, depth 0             768 MiB   = 768 x 1
                                 ----------
                                 99,461 MiB   against 101,170 measured
```

- `enc_ecd_x_copy` (residual; copied :562, depth 20 at :569, used :777, freed :790): its mod-switch at
  :777 is a no-op (`rtn` is already at depth 20).
- The bootstrap is already per ciphertext:

```cpp
for(int i = 0 ; i < 128 ; ++i)
    for(int j = 0 ; j < 6 ; ++j)
        bootstrapper.bootstrap_3(rtn[i*6+j], att_selfoutput[i*6+j]);
```

- Two bootstraps each move the pool by 42,239.9 MiB (agree to 0.1 MiB): 16,128 output plus
  26,111.9 = 768 x 34 MiB (a fresh ciphertext per call). Open: retention in `bootstrap_3` or
  freed-but-not-reused. Instrumenting `bootstrap_3`'s pool usage at entry and exit separates them (Phantom
  is compiled from source in this image).

## `s7_stream_boot4` refuted before measurement

Image built (`moai-gpu-s7_stream_boot4.sif`, digest 4bb55042..., job 46240178), not run:

```
after_gelu           81,120 MiB   +1,344
after_bootstrap3     81,984         +864
after_layernorm2     82,144         +160     <- the LAST growth event
after_bootstrap4     82,144           +0     <- the phase the step attacks
```

`patch_stream_boot4.py` assumed `s6` would move the binding phase to bootstrap 4; it moved to
LayerNorm2. `patch_chunk_ffn.py`'s header had already recorded that both sites reserve nothing.

```
growth event         reserved     used     gap      share of peak
keys_created          +40,576   40,498      78      49.4%   library row, dispositioned since s3
before_attention      +26,976  +11,520  15,534      32.8%   <- s7_fused_input_encrypt
after_attention        +2,624   +1,536  16,622
after_layernorm1       +9,600     +768  20,846      11.7%
after_gelu             +1,344   +6,144  25,262       1.6%
after_bootstrap3         +864     -768  26,126       1.1%
after_layernorm2         +160     +768  25,518       0.2%
after_bootstrap4           +0   +7,680  17,838
```

Maximum live set 64,306 MiB against a peak reservation of 82,144 MiB (21.7% fragmentation at that
point); the tail after `after_layernorm1` is 2,368 MiB.

## `s7_fused_input_encrypt` refuted by measurement

```
phase                s6 reserved   s7 reserved   s7 growth   s6 growth
keys_created              40,576        40,608     +40,608     +40,576
before_attention          67,552        52,608     +12,000     +26,976   <- THE TARGET
after_attention           70,176        70,432     +17,824      +2,624   <- the refill
after_layernorm1          79,776        79,328      +8,896      +9,600
after_gelu                81,120        81,120      +1,792      +1,344
after_bootstrap3          81,984        82,080        +960        +864
after_layernorm2          82,144        82,176         +96        +160
after_bootstrap4          82,144        82,176          +0          +0

peak_reserved_kb      84,115,456    84,148,224    +0.04%
peak_alloc_kb         77,650,579    77,650,579    IDENTICAL to the byte
wall                       3,399         3,393    -0.18%
LEVER|fused_input_encrypt|ciphertexts=768|parked=768|disk=1
```

- `before_attention` 67,552 -> 52,608 MiB (predicted 52,576: keys 40,498 + working array 11,520 + one
  full-level ciphertext).
- Transient levers on this pool, three measurements: `s3_input_copy_fused` (event 53,792 -> 43,136,
  peak unchanged), `s5_park_input_chunk_ffn` (event halved, peak -2.30%: bootstrap2, gelu, bootstrap3
  grew 26,208), this run (+0.04%). `s6_stream_boot23` lowered the peak by removing 768 full-level
  ciphertexts from the late phases' live set.

```
maximum live set (after_bootstrap4)   64,306 MiB
peak reservation                      82,176 MiB
gap                                   17,870 MiB = 21.7% of the published number
```

Earlier campaign's last rung (65.19 -> 60.97 GiB):

> s24 = s23 + gelu reloads on the thread's stream + slice 768: the per-ct default-stream syncs were
> the fragmentation source -- with them gone, the reserved landscape is flat at 61760 across
> boot2/FFN/GELU, i.e. the pool sits exactly at the audited floor (keys 40.5 + BSGS workspace ~12 +
> working set ~9).

This tree parks and reloads per ciphertext (`offload_cipher_to_host`, `reload_cipher_from_host`) with
`cudaDeviceSynchronize()` after the park loops on the default stream. The stream fix is derived from
this, not built.

## Earlier campaign (eighteen steps, other instrument; hypotheses only)

| lever shape | |
|---|---|
| stream a bootstrap or matmul output per ciphertext to host | |
| do a per-ciphertext operation earlier, fusing copy, mod-switch or offload into the producer | |
| free dead objects | |

| warning | finding |
|---|---|
| allocator levers | pool release threshold 0 crashed (illegal memory access in `cudaMallocAsync` during attention); trimming before each bootstrap ran and raised the peak (the reserved high-water grows) |
| parking | pays only when the consumer re-reads streamwise; wholesale reads refragment the pool |

## Off-path by construction

| candidate | reason |
|---|---|
| lower `logN` or fewer primes | security and precision trade |
| batch size | not a knob: batch is welded to the ring by slot packing (fewer batches empty slots, same ciphertexts) |
| unified-memory oversubscription | placement (VRAM spilled to host), prohibitive run time |
