# Measured overlay

The two files the online image is built from, copied from the build tree. The diff against pristine
(with reasons) is `../patches/online_from_pristine.diff`. `build_online.sbatch` copies this tree over
`/opt/EzPC/GPU-MPC/ext/sytorch/` in the parent image.

## `examples/bertbenchmark.cpp`

| change | reason |
|---|---|
| no `net.load("bert-tiny-weights.dat")`, `input.load("15469.dat", scale)` | files absent; bert-tiny sized at bert-base dimensions. Weights from each layer's seeded `_initScale` (identical on all three parties); input filled deterministically |
| input fill divides by 1000, not 1000000 | the old divisor put every input inside [-4, 4] at scale 12 (zero) |
| `GATE|logits|` | whole output as raw ring elements; evidence, not the gate (zeros, `off-path-runs/p_reveal/`) |
| `GATE|wire|` | the structural observable of both online parties; the gate |
| `n_layer` reads `LLAMA_N_LAYERS` | one-layer probes |
| `g_lazyWeights` | declared, assigned, never read (from dealer `d5`); kept to avoid a rebuild |

## `ext/llama/src/llama/comms.cpp`

| change | reason |
|---|---|
| `SocketBuf::read` transfers until complete | a freeze capture interrupting a large transfer killed the run; no protocol byte changed |
| `snni_wire_trace` | accumulates the gate over every received byte; order-dependent and order-independent accumulators |

Both are part of the measured program, declared in `../../MANIFEST.md`.
