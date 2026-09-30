# p_floor_bench: MOAI resident floor, measured

Floor micro-benchmark at T=1, VmRSS from `/proc/self/status` per component (counterpart of SHARK's
`ars_floor_bench`, SHAFT's `shaft_floor_bench`, BumbleBee's `floor_bench` in `bumblebee/off-path-runs/p_floor_bench/`). The target is MANIFEST.md
(streamed floor at chunk 128); this bench measures one key, a 14-key stage, both key banks and the
hidden-state vectors the stock code holds.

| | |
|---|---|
| patcher | `impl/p_floor_bench.py` (patches `test_full_scheme.hpp`, reuses the run's SEALContext, KeyGenerator, relin/gal keys, `gal_steps_vector`, Bootstrapper; returns before inference) |
| run | `SNNI_FLOOR_BENCH=1 OMP_NUM_THREADS=1` |
| build | job 46667103 (`normal`), `moai-cpu-p_floor_bench.sif` from `moai-cpu-m0_default.sif`, BUILD_OK |
| measure | job 46667236 (zen4 r19n02, T=1), wall 9:16, SLURM MaxRSS 179,169,892 kB = final VmRSS 178,772,624 kB |
| reproduce | `impl/p_floor_bench.py`, `impl/build_moai_floor.sbatch`, `impl/run_moai_floor.sbatch` |

```
after_context            3,655,692 kB   3.74 GB   SEAL context (overhead, excluded)
+ relin (+ public key)   1,438,524 kB   1.47 GB
+ gal_keys (30 bank)    38,719,172 kB  39.7 GB    1.32 GB/key
+ gal_keys_boot (47)    60,665,736 kB  60.7 GB    1.32 GB/key
largest stage, DIRECT   18,082,952 kB  18.5 GB    14 basicstep-32 keys built as their own set
+ 2 CT vectors @21limbs 55,462,924 kB  56.8 GB    resident
+ in-flight + 12 PS temp    466,944 kB   0.48 GB
```

- A key is 1.32 GB resident (= serialised size), identical across the 30-, 47- and 14-key builds.
- Two hidden-state vectors: live 2 x 768 x (2 x 21 x 65536 x 8) B = 33.82 GB, resident 56.8 GB (~23 GB
  pool retention from the mod-switch path).

Without streaming (live set):

```
keys, largest stage (15)   ~19.8 GB
relin                       ~1.5 GB
ciphertext payload (live)  ~33.8 GB
in-flight + PS temporaries  ~0.5 GB
-----------------------------------
total                      ~55.6 GB live (measured)
```

The bench's `FLOOR_SUMMARY` reports 76.7 GB (vectors resident with retention).
