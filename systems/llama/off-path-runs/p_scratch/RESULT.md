# p_scratch: leaked operator scratch

Jobs 45841981, 45842288, 45842339 (source introspection in the measured image), 2026-08-06. Became
the step `o2_operator_scratch_free` (first named `o3_operator_scratch_free` in the first ordering).

Table after `o2_activation_free` of the first ordering (peak 1,201,520 kB); weights (680.1 MB)
closed in `p_weights/`:

```
anon:[heap]                                        189.1 MB   0.157
heap:api.cpp:4120:SlothARS                          76.9 MB   0.064
heap:api.cpp:{2711,2712,2713,2714,2717}:SlothGelu   34.6 MB each  (173.0 MB together)
heap:keypack.h:363:DPFETKeyPack::DPFETKeyPack       31.5 MB   0.026
```

## Source

- `SlothGelu` (`api.cpp` 2707-2771): five `new GroupElement[size]`, no `delete[]`.
- `SlothARS` (4116-4165): one, no `delete[]`.

Audit of the 46 top-level functions in `api.cpp` that allocate with `new` (allocations against
releases up to the closing column-0 brace):

| | functions |
|---|---|
| release everything they allocate | 36 |
| release some (each hands something to its caller) | 6 |
| release nothing | 4: `SlothGelu`, `SlothSilu`, `SlothAttentionTriangular`, `SlothARS` |

- `SlothSilu` (SiLU) and `SlothAttentionTriangular` (causal attention) are not called by BERT and
  absent from the recorder table; left unpatched.
- Size: GeLU operand 128 x 3072 x 8 B = 3.146 MB per site; twelve layers never released ~37.7 MB;
  measured 34.6 MB (resident pages).
- `y`, `d`, `rp`, `abs`, `r` (`SlothGelu`) and `z` (`SlothARS`) are function-local, read before return,
  not returned or stored: freeing them changes no value.
- `anon:[heap]` (189.1 MB) is not an object; not a skip.

## Arm

Donor: SHAFT-CPU `s1_plaintext_free` (release past last use; also `s2_triple_share_free`, BumbleBee
`b2_result_ct_pack_free`). The closed arm borrows from every primary system (README.md, Two arms);
the provenance is in the step's `levers.env`.
