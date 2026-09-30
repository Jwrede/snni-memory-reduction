# s2_response_stream_on_ws: response stream on `s1_weight_stream`

Image `puma-s1_response_stream_v2.sif` (lever patches `distributed_impl.py` in the container) plus
the streaming driver: both levers. Off-path.

```
response_stream_v2 on s0_default        7,316,112 -> 6,406,676 kB   -12.4%
response_stream_v2 on s1_weight_stream  3,005,148 -> 3,061,920      +1.9%
                                        runs: 3,062,052 / 3,061,920 / 3,011,080
```

- The three concurrent 375,054,672 B `response` copies were a sixth of the 6.4 GB peak and do not set
  the 3.0 GB peak.
- `EQUIV|stream_vs_whole|max_abs=0` in all three runs.

## Repaired defect

`s2_weight_stream` was first published as a third row with `SIF=puma-s0_default.sif` (baseline image,
not cumulative). Passing `SIF` bypassed the guard in `puma_campaign.sbatch` ("a waterfall step may
not fall back to the baseline"). That run is a correct measurement against `s0_default` and became
`s1_weight_stream`; only this run was added.
