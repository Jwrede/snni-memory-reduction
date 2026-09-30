# p_seq197_faithful: sequence length 197

Diagnostic (other operating point; README.md, Transfer lines and sequence length). No model is
loaded: a geometry change. `BB_SEQ_LEN=197`, faithful operators, 12 layers, 16 threads per party,
3 replicates, PALMA `normal`.

| config | as | image |
|---|---|---|
| `f0` | `b0` | `bumblebee-f_b0_faithful_odd.sif` |
| `f7` | `b7`, 2 OT instances | `bumblebee-f_b4_faithful_odd.sif` |
| `f8` | `b8`, 1 OT instance | `bumblebee-f_p_tmo_b4_odd.sif` (`b4` tree, `BB_RECV_TIMEOUT_MS` from the environment, `TMO=1800000`) |

- `_odd` images fix the benchmark's `row_reduce` for odd row widths (not reached at 128).
- No recorder: with it, `f7` segfaulted in one party after ~55 s (139/134), `f8` aborted after ~72 s
  (`recorder_attempt/`). `run.sbatch` polls each party's `VmHWM` every 100 ms.

Maximum over parties, kB (party 0 heavier in every run):

```
          run1         run2         run3         median        wall (s)
f0   27,769,576   27,787,048   27,783,804   27,783,804    332 / 333 / 341
f7    8,314,944    8,324,856    8,308,076    8,314,944   1067 / 1062 / 1063
f8    8,588,656    8,532,904    8,567,468    8,567,468   1910 / 1879 / 1886
```

```
                         at 128      at 197
factor b0 / b8           10.60x      3.24x
factor b0 / b7            8.62x      3.34x
b8 against b7            -18.7%      +3.0%
```

- Endpoint grows 5.87x from 128 to 197, baseline 1.79x.
- All runs exit 0 on both parties, 151,296 gate values each. Rank-0 checksums: first layer 1534.00
  to 1535.10, final -7107.53 to -7104.30.
- Bytes sent by rank 0: `f0` 7,533,950,154 to 7,533,957,314; `f7` 7,524,334,840 to 7,524,349,504;
  `f8` 7,523,544,639 to 7,523,552,919 (`f8` against `f7` -0.01%).
