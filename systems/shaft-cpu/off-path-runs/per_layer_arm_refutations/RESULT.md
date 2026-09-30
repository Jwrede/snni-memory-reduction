# per_layer_arm_refutations: four per-layer variants, n=3 each

Levers built for the per-layer arm that did not become steps. `x6` and `x7` are the dispositions
`steps/s6_stream_load/levers.env` cites for its skipped object.

| row | base | base median (kB) |
|---|---|---:|
| `x3` | `x2_per_layer_disk` | 1,194,020 |
| `x5`, `x6`, `x7` | `x4_stream_load` (mechanism later published as `s6_stream_load`) | 906,020 |

```
                        replicates (kB)                      median      vs base   wall
x2_per_layer_disk   1,139,488 / 1,194,020 / 1,229,840      1,194,020        --      108 s
x3_low_cpu_mem_load 1,331,608 / 1,337,108 / 1,358,520      1,337,108     +12.0%     113 s
x4_stream_load        904,908 /   906,020 /   921,900        906,020        --      108 s
x5_graph_clear        904,156 / 1,035,528 / 1,050,956      1,035,528     +14.3%     108 s
x6_embed_stream       913,808 /   939,740 / 1,037,204        939,740      +3.7%     109 s
x7_param_defer        909,880 /   912,396 /   915,960        912,396      +0.7%     107 s
```

## x3_low_cpu_mem_load

| | |
|---|---|
| mechanism | `low_cpu_mem_usage=True`: module built on the meta device and filled from the checkpoint; no `state_dict` copy beside it |
| effect at the load | load high-water ~900 MB -> 21 MB; `after_model_load` 1,150,172 -> 300,084 kB |
| peak | +12.0%; the peak moves into the conversion phase |
| wall | +4.6% (113 against 108 s) |
| gate, transcript | unchanged |

## x5_graph_clear

| | |
|---|---|
| mechanism | `graph_clear_values_cpu`: drop each graph value after its last consumer |
| peak | +14.3%, replicates span 16.5% (allocator two-state spread) |
| earlier results of the same lever | +0.10%, +0.02% (allocator pinned) |
| earlier campaign | `05_eager_free` (CrypTen `_clear_unused_values()`): 140 MB worse |

## x6_embed_stream

| | |
|---|---|
| mechanism | `embed_stream_cpu`: embedding rows read from the spill file chunk by chunk |
| target | `heap:torch:torch.int64 held_by=frame`, rank 1 at `s5_per_layer`'s peak, 416.5 MB |
| peak | +3.7% |
| verdict | refuted: 14 fewer rounds on the same bytes and a moved gate; chunking should add rounds (`embed_chunk`: +30) |

```
              gate                                        transcript
x4_stream_load  -1.0620574951e+00  8.0404663086e-01       bytes=11228688384  rounds=1525
x6_embed_stream -1.0647888184e+00  8.1347656250e-01       bytes=11228688384  rounds=1511
```

## x7_param_defer

| | |
|---|---|
| mechanism | port of SHAFT-GPU's `param_defer`: CrypTen's scheduler materialises `Parameter` nodes last (GPU: -30.7% VRAM) |
| peak | 912,396 against 906,020 kB, spread 0.67% |
| verdict | broken program: logits ~1e+06 to ~1e+09 (fixed-point overflow or a share read at the wrong scale); transcript unchanged |
| cause | not established; candidate interaction with `encrypt_spill_cpu` (parameters reloaded in `Parameter.forward`) |

```
x4_stream_load  GATE|logits|-1.0620574951e+00  8.0404663086e-01
x7_param_defer  GATE|logits| 6.0271415000e+06 -3.8004904960e+09
```

## Summary

- `x5`, `x6`, `x7` attack objects live at the peak instant; none reduces it.
- The arm's working lever (`x4`, checkpoint streaming) removes an object that exists only because
  of how the program loads.
- The arm's peak is a 601,612 kB transient inside one `encrypt()` call, seen by the poller and named
  by no object table (basis of `s8_encrypt_chunk`).
