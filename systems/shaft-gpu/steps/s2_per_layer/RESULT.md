# s2_per_layer: transcript artefact evidence

Evidence file for `# transcript_artefact: 524288 RESULT.md` in `levers.env`. The per-layer driver
was first measured as `s5_per_layer` on top of `s4_param_defer` (earlier ordering); in the
published line it is `s2_per_layer`, admitted under the `transcript_artefact` rule of 2026-08-19.

## First measurement (earlier ordering)

```
                        VRAM reserved      host marker peak     wall
s4_param_defer               362,496 kB          2,924,220 kB    ~46 s
s5_per_layer run1            362,496             2,165,536        48 s
s5_per_layer run2            362,496             2,229,364        49 s
```

VRAM byte-identical; host marker peak about -24%. Host table at `s4_param_defer`'s peak
(`W_vram` 2.08, `W_host` 13.17, rule 3 selects the host):

```
870.5 MB  29.1%  heap:python3.10+0xd6be1
799.0 MB  26.7%  heap:libstdc++.so.6.0.29+0xf3269
433.3 MB  14.5%  file:db5325...          mapped, excluded from W by policy
366.8 MB  12.3%  anon:[heap]             located but unnamed, not targetable
```

The two rankable rows are one event: the ONNX conversion of the whole model.

## Transcript difference

```
s4_param_defer   bytes=11,229,212,672   rounds=1526
s5_per_layer     bytes=11,228,688,384   rounds=1525
```

524,288 bytes and one round fewer (0.0047%).

| hypothesis | test | result |
|---|---|---|
| mask built as zeros | `ExtendedMaskWrapper` traces `get_extended_attention_mask` as sub-graph 0 | 0 bytes, 0 rounds |
| a short sub-graph | per-sub-graph transcript | sums exactly; every layer identical |

```
   mask            0            0 rounds
   embed         420,519,936   60
   layer0..11    899,842,048  120   each, all twelve identical
   head           10,063,872   25
   -----------------------------------------
   sum        11,228,688,384 1525   = exactly what the run reports
```

Baselines without any lever:

```
shaft-cpu s0_default   bytes=11,228,688,384   rounds=1495
shaft-gpu s0_default   bytes=11,229,212,672   rounds=1496
```

## Location (2026-08-18)

Probe instrument `graphop_comm_gpu.py` (in this directory; loaded only in probe runs), reads
CrypTen's per-class counters after each graph:

```
                    p_comm_s4                p_comm_s5
   total       11,229,212,672  1526      11,228,688,384  1525
   softmax      4,605,345,792   492       4,605,345,792   492
   gelu         4,454,350,848   228       4,454,350,848   228
   matmul       1,632,681,984    98       1,632,681,984    98
   embedding      415,719,424    34         415,719,424    34
   layernorm      120,012,800   650         120,012,800   650
   tanh               577,536    23             577,536    23
   conv                     0     0                   0     0
   other              524,288     1                   0     0
```

- The whole difference is in `other`: one node of the extended-attention-mask chain under `bert.`
  (`ConstantOfShape`, `Equal`, `Expand`, `Where`, `Where_1`), encrypted in the GPU whole-model
  export, local arithmetic in the decomposed graph.
- It predates every lever on both systems and belongs to the export: a removal, declared with its
  exact byte count.
- Rejected precedent of a different kind: BumbleBee's `x7_nonlinear_chunk` (+41% bytes).

Instrument versions: v1 read `crypten.communicator.get()` at import (before `crypten.init()`, run
died after 6 s); v2 measured 36,864 bytes because `Graph.forward` calls
`reset_communication_stats()` before every node; v3 (above) agrees with the gate's transcript to
the byte.
