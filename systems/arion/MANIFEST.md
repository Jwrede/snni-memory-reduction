# ARION: manifest

| field | value |
|---|---|
| system | ARION (Lattigo/Go), non-interactive RNS-CKKS |
| arm | closed: donor levers measured on primary systems only (README.md, Two arms); outcome transfer, no-op or regression |
| machine | PALMA `bigsmp` (72-core Skylake AVX-512, 1.5 TB, `--exclusive`) |
| workload | BERT-base, logN=16, rows=128, batch=256, two transformer layers, declared (menu option 12) |
| scheme | RNS-CKKS, N=2^16, LogQ = [58, 45 x 14], LogP = [60 x 4], scale 2^45, MaxSlots 2^15 |
| image | `sif/arion-<step>.sif`, built by `impl/build_sif.sbatch` from `impl/arion.def` |
| upstream | `github.com/hangenba/Arion`, lattigo v6.1.1, binary md5 `a8a1fd50b7197993f2319e63021eb254` |
| pools | host only, single process |
| replicates | 1 from 2026-08-05 (the baseline keeps its three; the gate tolerance rests on them). The Go heap census needs a whole `bigsmp` node (1.5 TB requested, ~600 GB written transiently), shared with MOAI-CPU. Reasoning: `tools/palma/conf/arion.conf`; each row carries `n_runs` |
| threads | `ARION_THREADS=64` at the baseline (upstream's hardcoded value, now a runtime parameter) |

## Build

```
impl_upstream: https://github.com/hangenba/Arion.git
impl_commit: bd5fcd7f29af2f35c954dc20c83accdc6b8e2a14
impl_instrument: pin_workload.py
impl_build_apptainer: apptainer build arion-<step_id>.sif impl/arion-<lever>.def
impl_build_palma: sbatch --export=ALL,STEP=<step_id> impl/build_sif.sbatch
```

- Commit also recorded in `/opt/arion_commit.txt` in the image.
- `impl/levers/` holds lever files of every ordering measured, named for the ordering they were
  written for. The published order is `steps.csv`.

## Target

```
target_host_kb: 8704000
overhead_host_kb: measured_per_run
replicates: 1
gate_tier: T2
gate_tolerance_abs: 1.5e-3
gate_observable: the decrypted per-layer output matrices the PRISTINE program already writes,
                 Output/<model>/layer_<i>_output.csv for every layer that ran, flattened to one
                 value per line into logs/gate.txt and announced as
                 `GATE|layer_outputs|files=N|n=..` plus `GATE_NONZERO|nz/n`
seed: NOT PINNED, and deliberately not: the CKKS keys come from lattigo's own key generator, which
      draws from crypto/rand and offers no seed hook, so pinning it would mean patching a third
      party's cryptography and measuring a program nobody else can run. The consequence is
      declared instead of hidden: the gate is T2 with a tolerance derived from the scheme's
      noise, and whether that suffices is settled by the two-run check and recorded in README.md.
```

Rule (as MOAI-CPU): the streamed floor, the maximum over a layer's instants once every streamable term
is streamed; one thread; library context excluded.

Sizes (Lattigo v6.1.1, parameter literal `configs/config.go:327-351`, probe in the campaign image):

| unit | size |
|---|---|
| ciphertext at level L | (L+1) MiB |
| residual key-switching key | 4 digits x 2 x 19 limbs x 65536 x 8 B = 77,824 kB |
| bootstrapping key | 8 digits x 2 x 34 limbs x 65536 x 8 B = 278,528 kB |
| attention (`ComputeAttentionMT3`) | 31 rotations, 30 distinct Galois elements |
| bootstrap | 48 distinct elements; largest BSGS stage (SlotsToCoeffs stage 0) 14 keys |
| check against runs | stock keys 128 x 79.7 MB + 48 x 285.2 MB = 23.89 GB (measured row 23.9 GB); minimal keys 2.39 + 13.69 = 16.08 GB (measured 16.1 GB) |

| instant | live under streaming | kB | GB |
|---|---|---|---|
| **QK^T in a head** | 30 attention keys 2,334,720 + relin 77,824 + Q 64@13 917,504 + K 64@13 917,504 + rotated Q and K blocks 2 x 917,504 + scores 128@12 1,703,936 + V 64@13 917,504 | **8,704,000** | **8.91** |
| Exp x V in a head | 21 keys + relin + V + rotated V + exp scores + accumulators (upper bound) | <=6,955,008 | <=7.12 |
| softmax bootstrap in a head | boot stage 14 keys 3,899,392 + btp relin 278,528 + exp scores + V (upper bound) | <=6,866,944 | <=7.03 |
| bootstrap 1..4, chunk 1 (each worker bootstraps one ct with its own evaluator) | boot stage 3,899,392 + btp relin 278,528 + output, residual, real, imag ct | 4,245,504 | 4.35 |
| matmuls, GELU, LayerNorm | chunked, no key in the matmuls | chunk-dependent | < 0.5 |

- Streamed, not counted: layer input and residual (768 ct at level 14), FFN intermediate (3072 ct),
  other heads' Q/K/V and finished outputs, key banks outside the operation or BSGS stage in flight,
  encoded DFT diagonals (public constants).
- QK^T does not chunk: Q and K of the head feed all 128 score diagonals, needed together for the
  softmax normalisation. ARION rotates both operands (Q by giant, K by baby steps): one rotated copy
  of each in flight.
- Convention (thesis App. B.4, "no derived copy held beside its source"): holding the ten
  giant-rotated Q blocks instead gives 16,961,536 kB (17.4 GB). Finer conventions: 11 baby keys plus
  the current giant pair, 7,380,992 kB; additionally streaming Q and K by column moves the maximum to
  the bootstrap instant (~4.35 GB).
- Measured: `off-path-runs/p_floor_bench_att`, the QK^T set at T=1: 8,649,616 kB above the context
  (-0.62%); keys alone 2,348,156 against 2,334,720 kB derived.
- Full derivation: `~/master-thesis/ARION_TARGET_DERIVATION.md`; probe sources on PALMA under
  `/scratch/tmp/j_wred02/arion_target_probe/`.

## Two layers

| | |
|---|---|
| reason | MOAI-CPU, the paradigm partner, takes ~144 h per run at this depth; the closed arm compares the two at one depth |
| earlier campaign | no 12-layer run completed: OOM-killed in layer 0 on a 251 GiB machine (working-set floor at least 242 GiB) |
| recorded in | RUN_META and the program's `WORKLOAD|` marker, on every row |

## Gate

| | |
|---|---|
| observable | per-layer outputs the pristine program writes (`Output/<model>/layer_<i>_output.csv`); nothing added to the computation |
| first declaration | `gate_tolerance: 1e-6` relative; the three baseline replicates exceeded it and `make_steps.py` refused the gate |
| why absolute | 96.1% of the slots are structural zeros of the packing (median |v| = 5.9e-9, max ~11) |
| replicates | not bit-identical (keys drawn fresh per run); reproducible within the bound |

Bound, from the source (derived after the failure, as on BOLT):

```
unit:   worst-slot noise of ONE lattigo v6.1.1 default bootstrap. configs/config.go:341ff
        overrides only LogN/LogP/Xs, so the circuit is the library default, whose documented
        mean precision is ~27 bits; the worst slot of 2^15 sits a few bits under it -> ~2^-23.
gain:   the segment between the last bootstrap and the observable, worst case coherent:
        a 768-column matmul row-sum (2^9.6) times the InvSqrt local gain at its declared domain
        floor (InvSqrtMinValue2 = 0.10 -> derivative of x^-1/2 there ~ 2^4); GELU's Chebyshev
        evaluation is O(1). Segments further back are renormalised by the next LayerNorm.
bound:  2^-23 x 2^9.6 x 2^4 = 2^-9.4 ~= 1.5e-3 absolute.
```

| check | result |
|---|---|
| depth-flatness | worst pairwise disagreement over the three replicates 3.8e-4 (layer 0), 3.0e-4 (layer 1); inside the bound by 3.9x; accumulating error would grow with depth |
| slot classes | structural-zero slots differ by at most 5.2e-8 (~2^-24.2, two bootstraps' additive floor); value-carrying slots value-scaled (<= 3.3e-5 relative) |

1.5e-3 absolute on values up to ~11 catches structural changes, not changes confined to the last bits.
