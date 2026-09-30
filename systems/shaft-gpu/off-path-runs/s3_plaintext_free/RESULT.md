# s3_plaintext_free: not a step

Jobs 46145953-55, `gpuh200`, n=3, rc=0. Earlier ordering, base `s2_embed_chunk`.

```
PEAK DID NOT MOVE s3_plaintext_free: host rise 2568 kB is 0.09% of the peak, BELOW this
step's own run-to-run spread of 5.398% over 3 runs
GATE FAILED s3_plaintext_free: output differs from the baseline, worst relative difference
1.000e+00 against tolerance 1.000e-06
```

| | |
|---|---|
| object | plaintext model, 972 MB, 33% of the peak, 171 float32 tensors in an `OrderedDict` (Python census at rss 2,944,104 kB, 1.62 GB over 225 objects; probe directory not retained) |
| lever | `plaintext_free_gpu.py` releases it after the shares are derived |
| marker-phase peak | 2,557,584 -> 2,379,980 kB, -6.9% |
| published peak (`VmHWM`) | unchanged, +0.09% |
| VRAM | byte-identical in all runs |
| gate | failure inherited from `s2_embed_chunk` (phase B) |

- A first report of -19.3% compared `host_marker_peak_kb` with the published `VmHWM` peak (two
  different quantities); `make_steps.py` refused it.
- The high-water is reached while model and shares are both live; a later release cannot lower it.
- Next candidate named at the time: the `torch.int64` shares (653 MB), protocol data (placement,
  not release).
