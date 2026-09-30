# MOAI-GPU: ciphertext level audit

Result: the levels match the source and predict the measured VRAM to the MiB. Settled 2026-07-30 from
existing artifacts (no cluster time).

## Why levels matter for memory

```
2 polynomials x (level + 1) limbs x N coefficients x 8 B
```

N = 65536: one limb = 2 x 65536 x 8 = 1,048,576 B = 1 MiB per level per ciphertext. Arrays are 768
ciphertexts, so one level off on one array is 768 MiB. A scale bug can leave ciphertexts at the wrong
level, i.e. the wrong size.

## Source

- Level trace: printed by the pristine program at 16 phase boundaries (`test_single_layer.cuh`);
  captured in `SNNI-Benchmark/canonical_runs/results/moai_gpu_waterfall/s0_default/stdout.log` with
  pool occupancy in `markers.log` (pristine plus markers, no levers, per its `config.txt`).
- `chain_depth()` = `total_coeff_modulus_.size() - 1` (`phantom-fhe/include/context.cuh:129`): depth
  `d` = `d + 1` limbs = `(d + 1) MiB`.

| phase | observed | source |
|---|---|---|
| before attention | 14 | fresh at 34, then `boot_level + (remaining_level - remaining_level_att)` = `14 + 5` switches, then one more (:565-588). 34 - 19 - 1 = 14 |
| after selfoutput | 0 | attention plus the self-output matmul consume the remaining 14 |
| after bootstrap 1 | 20 | bootstrap restores to `remaining_level` = 20 (:394) |
| after layernorm 1 | 0 | LayerNorm consumes all 20 |
| after bootstrap 2 | 20 | restores to 20 |
| before intermediate linear | 9 | eleven explicit `mod_switch_to_next_inplace` immediately after boot 2 (:832-836). 20 - 11 = 9 |
| after inter layer | 8 | one plaintext multiply costs one level |
| gelu | 1 | the Chebyshev GELU consumes 7 |
| after final layer | 0 | one plaintext multiply |
| after bootstrap 3 | 20 | restores to 20 |
| after layernorm 2 | 0 | LayerNorm consumes all 20 |
| after bootstrap 4 | 20 | restores to 20 |

## Memory predicted from levels

At `inputs_prepared`: `enc_ecd_x` at depth 14 (:587, 15 MiB each) and `enc_ecd_x_copy` at depth 20
(:565-569, 21 MiB each), 768 each:

```
predicted   768 x (15 + 21) MiB = 27,648 MiB
measured    pool_used 68,146.1 - 40,498.1 = 27,648.0 MiB
```

Two full bootstraps from comparable states:

```
after_selfoutput -> after_bootstrap1   58,930.1 -> 101,170   = +42,239.9 MiB
after_final      -> after_bootstrap3   72,754.1 -> 114,994   = +42,239.9 MiB
```

768 x 21 MiB output plus a residual of 768 x 34 MiB (later shown to be fragmentation:
`off-path-runs/p_bootprobe/`).

Key bank: 31 Galois keys plus 1 relin key at `35 components x 2 x 36 limbs x 65536 x 8 B` = 40,325 MiB
from source against 40,498.1 MiB at `keys_created` (0.4%: public key and NTT tables). Modulus chain
from the program's printout: `51 + 20 x 46 + 14 x 51 + 58`, 36 primes, `poly_modulus_degree 65536`.

## Scope

- The VRAM footprint is that of a program whose ciphertexts sit at the prescribed levels.
- The output is numerically invalid (values, not levels): wrong rotation index, coefficients, BSGS
  composition or overwritten scale leave levels intact.
- A lever that changes when a mod-switch happens changes sizes; its `levers.env` cites the level trace.
