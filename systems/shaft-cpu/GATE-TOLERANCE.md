# SHAFT-CPU: gate tolerance, measured

Diagnostic runs, not steps. Probe `impl/levers/probe_seed_sweep_cpu.py` under
`stepenv/p_seedsweep_<offset>.env`.

| sweep | jobs | date | runs |
|---|---|---|---|
| all seeds, 12 layers | 45514843-47, 45515406-07 | 2026-07-30 | `evidence/seed-sweep-20260730/` |
| all seeds, 1 layer (`SHAFT_N_LAYERS=1`) | 45519516, 45519556-61 | 2026-07-30 | same |
| masks only (`SHAFT_SEED_SCOPE=masks`), 12 and 1 layer | 47060584-95 | 2026-09-29 | `evidence/seed-sweep-masks-20260929/` |

## Resolution against tolerance

| | |
|---|---|
| declared | `gate_tier: T2`, `gate_tolerance: 1e-6` relative |
| format | CrypTen fixed point, 16 fractional bits; published logits 8836/65536 and -15160/65536 |
| 1 ulp | 2^-16 = 1.526e-5 absolute = 1.1e-4 relative on logit 0, 113x the tolerance |
| consequence | the gate admits only byte-identical output (operationally T1) |
| truncation | 2-party `div_` truncates each share locally (`mpc/primitives/arithmetic.py:462-469`); the last bits depend on the masks |

## Sweep 1: all seeds shifted, 12 layers

Shift preserves the per-rank offsets and the shared global seed. Markers (`all_markers.txt`):

```
SEEDSWEEP|crypten|offset=0x1111|next=0xdeadd000|local=0xc110ff|global=0x6ffe    (party 0)
SEEDSWEEP|crypten|offset=0x1111|next=0xdeadd001|local=0xc11100|global=0x6ffe    (party 1)
SEEDSWEEP|torch|offset=0x1111|seed=0x6ffe
```

| seeds | logit 0 | logit 1 | argmax |
|---|---:|---:|:--:|
| published | 0.13482666 | -0.23132324 | 0 |
| +0x1111 | 0.00268555 | -0.05754089 | 0 |
| +0x2222 | 1.15917969 | -0.88706970 | 0 |
| +0x3333 | 0.92845154 | -0.67521667 | 0 |
| +0x4444 | -1.18324280 | 0.72396851 | 1 |
| +0x5555 | -0.80229187 | 0.50646973 | 1 |

- Three runs at `+0x1111` are byte-identical: the variation is the seed.
- Logit 0 spans 2.34 against 0.135 (~153,000 ulps, 17x the value). The label flips in 2 of 5.

## Sweep 2: all seeds shifted, 1 layer

| seeds | logit 0 | logit 1 | argmax |
|---|---:|---:|:--:|
| published | 0.14779663 | 0.17158508 | 1 |
| +0x1111 | 0.10717773 | 0.13999939 | 1 |
| +0x2222 | 0.02214050 | 0.27108765 | 1 |
| +0x3333 | 0.31022644 | 0.12890625 | 0 |
| +0x4444 | 0.04896545 | 0.30772400 | 1 |
| +0x5555 | 0.00563049 | 0.16517639 | 1 |

- Repeat at `+0x1111` byte-identical.
- Logit 0 spans 0.3046 against 0.148 (ratio 2.06 against 17.4 at 12 layers). The label flips in
  1 of 5.
- BumbleBee comparison: at N=1 two unchanged runs agree to 6.0e-6 relative, and a mechanism-derived
  bound (7.3e-3) contains the observed 5.07e-3. Here the same depth moves the output by 3e-1.

## Sweep 3: masks only, valid input (2026-09-29)

Input token IDs drawn from 0..28995 (cased vocabulary; before: 0..30521). Only the CrypTen seeds
(masks and triples) are shifted; torch seed and input unchanged.

| seeds | 12 layers: logit 0 | logit 1 | argmax | 1 layer: logit 0 | logit 1 | argmax |
|---|---:|---:|:--:|---:|---:|:--:|
| published | 0.55476379 | -0.06614685 | 0 | 0.08537292 | 0.24484253 | 1 |
| +0x1111 | -2.72485352 | 2.16091919 | 1 | 0.07992554 | 0.25299072 | 1 |
| +0x2222 | 0.42149353 | -0.16069031 | 0 | 0.08547974 | 0.26985168 | 1 |
| +0x3333 | 1.14962769 | -0.75050354 | 0 | 0.08682251 | 0.26110840 | 1 |
| +0x4444 | -2.69200134 | 2.14460754 | 1 | 0.10676575 | 0.26237488 | 1 |
| +0x5555 | -2.08032227 | 1.61184692 | 1 | 0.12086487 | 0.27212524 | 1 |

- 12 layers: label flips in 3 of 5; logit 0 spans 3.87 against 0.555.
- 1 layer: label stable; logit 0 moves up to 0.035 absolute (~2,300 ulps), logit 1 up to 0.027.

## Consequences

| | |
|---|---|
| tolerance | kept at 1e-6; a tolerance covering the mask spread (~2.3 absolute at 12 layers, at least ~0.3 at 1 layer) admits any output |
| T3 (label) | not usable: the label moves with the masks |
| draw-changing levers | cannot pass the output gate; admitted only as phase B (README.md rule 7): same exchanged bytes plus a plaintext equivalence test |
| phase B steps | `s5_per_layer`, `s6_stream_load`, `s7_embed_chunk`, `s8_encrypt_chunk`; tests `impl/equiv_*.py` |
| off-path | levers that change the computed function (polynomial GELU) |
| earlier campaign's `02_per_layer` | unseeded (`ct.init()` without seeding; CrypTen seeds from `os.urandom(8)`), no baseline comparison, no plaintext test; not verifiable after the fact |
| N=1 gate beside an N=12 peak | not taken: `make_steps.py` collapses disagreeing replicates to `NONDETERMINISTIC`; phase B covers the case |

Five seeds are a sample, not a bound.

Same pattern on BumbleBee: two unchanged 12-layer runs differ by 3.4x on the output checksum, at
N=1 by 6.0e-6 relative.
