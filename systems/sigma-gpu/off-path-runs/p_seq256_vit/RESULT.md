# p_seq256_vit: sequence length 256

Diagnostic (README.md, Transfer lines and sequence length).

| | |
|---|---|
| obstacle | `keyBufSz = 20 * OneGB` (`experiments/sigma/sigma.cu`, bert-base branch): a model-size table (bert-tiny 1 GB, bert-base 20, bert-large 50); gpt2 scales by integer division (same value for 192, 196, 197). `p_bufhighwater`: 16.83 of 20.00 GiB used (84.2%) at 128 tokens |
| patch | `impl/patches/patch_keybuf_env.py`: settable via `SIGMA_KEYBUF_MB`, inert otherwise, at the point of use; each run prints `SNNI_KEYBUF|source=env|bytes=..` |
| sizes | buffer 52,854 MB at the baseline, window 3,179 MB at the endpoint (from the measurement, the window was sized from `p_keywindow_high`'s `op_high = 825,125,000` at 128) |

Party 0, `VmHWM`:

```
                                  run1        run2        run3      median     rc
p_seq256_p_vit_s0           57,324,720  67,809,856  67,809,724  67,809,790   137/0/0
p_seq256_p_vit_s6            5,395,908   4,456,900   4,485,648   4,485,648   0/0/0

                            factor 15.117x at n_seq=256,  against 16.13x at n_seq=128
```

- `s0` run1 oom-killed (`--mem=110G` against two parties each holding 51.6 GiB; node 1,547 GB); its
  partial trace reached 57.3 GB.
- Endpoint spread 20.9% (5,395,908 against 4,456,900).

Maximum over parties (from `party_peaks.txt`): baseline 67,811,944 / 67,809,772 kB (median
67,810,858), endpoint 5,397,044 / 4,457,012 / 4,487,716 kB (median 4,487,716): 15.110x at 256 against
16.130x at 128 (94%).

197 still fails for a second reason that admits 128 and 256 and refuses 192, 196, 197, 198; the
comparison is labelled at 256. A power-of-two requirement is consistent with this, not established.
