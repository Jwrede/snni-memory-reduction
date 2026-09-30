# per_layer_arm: per-layer driver, three steps

Not rows of the published line: a different measured script (`SHAFT_SCRIPT=bert_perlayer_cpu.py`),
and the script md5 identifies the program. n=3 each. Measured before the 2026-09-29
re-measurement; the mechanism was later published as `s5_per_layer` and `s6_stream_load`.

## Rows

```
                       leader peaks (kB)                     median      spread   wall
x0_per_layer_base  1,986,180 / 2,031,780 / 2,030,472       2,030,472    2.25%    108 s
x1_per_layer       1,420,492 / 1,420,316 / 1,417,756       1,420,316    0.19%    106 s   -30.1%
x2_per_layer_disk  1,133,832 / 1,259,112 / 1,139,484       1,139,484   11.0%     108 s   -19.8%
```

| row | levers |
|---|---|
| `x0_per_layer_base` | per-layer driver, no lever |
| `x1_per_layer` | the five levers of `s4_onnx_and_spill` plus `embed_chunk_cpu` |
| `x2_per_layer_disk` | `x1` plus the earlier campaign's `03_disk_offload` |

`x2`'s spread: two replicates at ~1.135 M, one at 1.259 M kB, the two-state glibc behaviour of
`off-path-runs/s5_graph_clear_values/`. The effect is twice the spread.

## Against the line at the time (published quantity, kB)

```
s0_default (whole model, no lever)     3,575,304
x0_per_layer_base (decomposed, none)   2,026,372      -43.3%
s5_embed_chunk (whole model, 6 levers) 2,056,996      -42.5%
x1_per_layer (decomposed, 6 levers)    1,416,216      -60.4%   2.525x
x2_per_layer_disk (+ layers on disk)   1,135,384      -68.2%   3.145x
```

- Driver and levers are nearly independent: 2.53x together, 1.76x and 1.74x apart.
- The whole-model peak is the ONNX conversion; three releases in that phase each freed 433 MB and
  moved the peak by nothing (`off-path-runs/s6_conversion_phase/`). The decomposition never converts
  the whole model.
- Earlier campaign, for reference: `02_per_layer` 1,834 MB against a 3,428 MB baseline, best variant
  `04_reduce_temp` 1,294 MB; no gate, transcript or plaintext test.

## Phase B evidence

Per-layer conversion interleaves layer i+1's PRZS draws with layer i's Beaver draws, so the output
changes.

```
s0_default (whole model)   bytes=11,228,688,384   rounds=1495
x0_per_layer_base          bytes=11,228,688,384   rounds=1495   subgraphs=14
x1_per_layer               bytes=11,228,688,384   rounds=1525   subgraphs=14
```

- The decomposition alone is transcript-identical. `x1`'s +30 rounds are `embed_chunk_cpu`'s 16
  chunked invocations (same +30 as `s5_embed_chunk` and SHAFT-GPU's port).
- `impl/equiv_per_layer.py` on this port:

```
ok  extended mask for an all-ones attention_mask is all zeros, shape (1, 1, 1, 128)
ok  12 layers: decomposition is bit-identical to the whole model
PASS
```

The zero-mask assumption is checked against the model's `get_extended_attention_mask`; with padding
it would not hold.

## Gate

```
s0_default          1.3482666016e-01  -2.3132324219e-01   -> class 0
x0_per_layer_base  -1.0504302979e+00   8.0657958984e-01   -> class 1
x1_per_layer       -1.0620574951e+00   8.0404663086e-01   -> class 1
```

- The class differs between the two programs. `GATE-TOLERANCE.md`: at 12 layers a different mask
  stream moves this observable by 2.34; at 1 layer the argmax flips for 1 seed in 5.
- Within the arm the rows agree to 1.1%.

## x2: disk offload

| | |
|---|---|
| mechanism | each encoder layer's `state_dict()` written to disk at startup, rebuilt one layer at a time, file removed after read; 340,278,048 B, 12 layers |
| relation | additive to `encrypt_spill_cpu` (which spills encrypted shares) |
| allocator | `malloc_trim(0)` after each `gc.collect()` is part of the lever; the donor's `MALLOC_ARENA_MAX=2` is not taken (global allocator knob) |
| gate fix | `BertLayer(config)` draws random init weights from torch's global generator before `load_state_dict`; construction wrapped in `torch.random.fork_rng()` |
| gate | byte-identical to `x1`: no phase B needed |

```
x1_per_layer       GATE|logits|-1.0620574951e+00 8.0404663086e-01
x2_per_layer_disk  GATE|logits|-1.0620574951e+00 8.0404663086e-01
```

## Not admissible

| lever | reason |
|---|---|
| earlier campaign's `04_reduce_temp` | polynomial GELU in place of the Fourier series (192 MB -> 12 MB per call, max error 0.0041 against 0.0046, 19% faster): different function and transcript, an approximation choice |

The earlier campaign's `05_eager_free` (CrypTen `_clear_unused_values()`) was 140 MB worse; this
campaign's `graph_clear_values_cpu` gave 0.02%.
