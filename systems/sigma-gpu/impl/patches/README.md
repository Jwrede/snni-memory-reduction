# SIGMA-GPU patches

## gate_deterministic_io.diff

Setup, not a lever: no memory behaviour changes; applied to the baseline and every step.

```
git apply -p1 gate_deterministic_io.diff       # verified with --check against pristine source
```

(from the `EzPC` source root)

| defect in the stock program | effect | fix |
|---|---|---|
| `input.zero()` and `net->zero()` | output zero for arithmetic reasons; the earlier campaign read `0, 0` as correctness | input and weights filled with a fixed arithmetic sequence |
| `srand(time(NULL))` (`sigma.cu:170`) | no run reproducible | `srand(0x5EED)`, `SEEDED|` marker on stderr |
| observable was argmax plus one value | close to a label (T3) | full decrypted output vector as `GATE|logits|` (the line existed, commented out) |

| choice | reason |
|---|---|
| fill once after the per-model `init`/`zero` block | one edit point for gpt2, bert-tiny, bert-base, bert-large, llama |
| `den=1000` (input ~+-1.0), `den=28000` (weights ~+-0.036 = `1/sqrt(768)`) | twelve layers must not wrap the 50-bit ring |
| `GATE_NONZERO|<nz>/<n>`, `GATE_DEGENERATE|` | the harness fails a run with an all-zero output |

- Expected invariants under the patch (FSS is data-oblivious): peak unchanged, total communication
  1,062,390,674 B.
- First version divided by 1000000: values in `[-5, 4]` at `1 << 12 = 4096`, output 98,304 zeros,
  exit 0, well-formed `GATE|logits|` line; it would have passed the two-run check.
- Measured result (`off-path-runs/p_det/`): seed and fill verified, output still 98,304 zeros; the gate
  is the transcript (`T1s`, `MANIFEST.md`).
- Open: whether FSS key generation draws from the C library RNG (no `rand()` found in the sigma
  experiment or its backend header). Settled by repeated runs (README.md, Seeds).
