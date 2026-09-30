# m27_boot_chunk_64: bootstrap chunk 64

Off-path: the target is taken at the bootstrap chunk of 128 introduced by `m8` (MANIFEST.md; README.md
rule 6); chunk 64 reduces a target term (the chunk in flight). On `m26_att_keys_late_create`, image
`moai-cpu-m27z_boot_chunk_64.sif`, markers `LEVER|boot_chunk_64|size=64` and
`LEVER|chunked_bootstrap|size=64` once per layer.

| run | directory | VmHWM (kB) | wall (s) | against `m26` (54,015,472 kB) |
|---|---|---:|---:|---:|
| `m27z` | `logs_m27z/` | 52,016,560 (polled) | 100,022 | -3.7% |
| `m27z2` | `logs_m27z2/` | 49,965,052 | 103,846 | -7.5% |

- Shared Zen 4 node, both rc=0, gate identical to `m26`.
- `m26`'s other run (`m26z`, 53,507,460 kB) puts `m27z` 2.8% below it. The two `m27` runs differ by 4%
  (node spread 2%); pool retention 10.3 GB in `m27z2`. Only `m27z2` has an object table.
- Bootstrap windows fall (layer 0 bootstrap 1: 46.3 GiB in `m27z` against 51.0 in `m26z`); the peak
  moves to the second layer norm (49.4 / 49.6 GiB in `m27z`, 49.3 in `m26z`).

At that peak (`m27z2`, `objects_host.jsonl`):

```
16.91 GB  layernorm.hpp:410        layernorm2 output array (768 ciphertexts)
16.91 GB  test_full_scheme.hpp:851 bootstrap output, the layer norm's input (768 ciphertexts)
10.30 GB  sealpool:retained        freed blocks below the pool threshold
 3.48 GB  SEALContext
 1.61 GB  test_full_scheme.hpp:1404 plaintext-matmul copy
 1.32 GB  relinearization key
```

Both arrays are live; chunking the layer norm would be next (not measured).
