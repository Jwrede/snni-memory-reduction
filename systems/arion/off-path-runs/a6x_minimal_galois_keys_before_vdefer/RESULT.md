# a6x_minimal_galois_keys_before_vdefer

| | |
|---|---|
| run | 47044192, shared `bigsmp` (18 cpus, 700 GB), one run, rc 0, submitted 2026-09-29 before a5 was measured |
| image | `arion-a6_minimal_galois_keys_goheap.sif` = `a5_qkv_head_release` + `patch_minimal_galois_keys.py`, ARION_THREADS 16, GOGC 25 |

```
a5_qkv_head_release       131,619,236 kB
this run (a5 + keys)      126,358,852 kB   -4.0 %
```

Not the line's a6: at a5's peak `matrix.RotateCiphertextMatricesHoistingMT.func1` (32.9 GB) lies above
`he.ParallelGenGaloisKeys.func1` (23.9 GB) and has a free fix (rotated V blocks created before Step 1,
read only in Step 5 of `ComputeAttentionMT3`): a6 is `a6_v_rotation_defer`, the keys follow as a7.

## Key row (`census/objects_go.jsonl.gz`)

- 100% 512 KiB limb arrays (65,536 x 8 B) from `rlwe.KeyGenerator.GenGaloisKeyNew`: 45,560 limbs at a5,
  30,665 here.
- `ParallelGenGaloisKeys` serves the attention keys (`GenerateKeysAndBtsKeysMT`, `keysMT.go:57`) and
  the bootstrapping rotation keys (`GenEvaluationKeysParallel`, `keysMT.go:200`); the goroutine stack
  has no caller frame, so both land on one row.
- Stock code requests 256 rotations (+-i*256, i = 1..128); at 32,768 slots +i*256 and -(128-i)*256
  coincide: 128 distinct Galois elements resident.
- One attention key: ceil(15/4) = 4 digits x 2 x 19 limbs = 152 limbs = 79.7 MB. The lever keeps 31
  rotations, 30 distinct (-10*3072 coincides with baby step 8*256).

```
removed:   (128 - 30) x 152 = 14,896 limbs   measured 45,560 - 30,665 = 14,895 limbs   (7.8 GB)
remainder: 45,560 - 128 x 152 = 26,104 limbs at a5, 30,665 - 30 x 152 = 26,105 here   (13.7 GB)
```

- The remainder is the bootstrapping rotation key set (required). Peak -5.3 GB of the 7.8 GB removed.
- Predicted -18 to -21 GB (assumed the whole row was attention keys).
- Cost declared by mechanism (fewer keys generated). `a3_minimal_galois_keys` read +22.4% wall in one
  run against one run; this run's wall is shared-node.
