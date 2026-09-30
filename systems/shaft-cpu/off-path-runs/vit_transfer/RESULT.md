# vit_transfer: SHAFT-CPU on ViT-Base/16

Transfer to another model (README.md, "Transfer lines and sequence length"). SHAFT-CPU loads and
traces the real model. Published as the system `systems/shaft-cpu-vit`.

| | |
|---|---|
| runs | `vit_v0_default` (unmodified driver), `vit_v1_endpoint` (BERT endpoint's lever stack) |
| `vit_v1` levers | `SHAFT_MODEL=/vit SHAFT_EMBED_CHUNKS=16 SHAFT_ENC_CHUNKS=8` plus per-layer streaming |
| replicates | 3 each, all rc=0 |
| machine, instrument | as the BERT line |

```
                  leader VmHWM per replicate (kB)              median        spread
vit_v0_default    3,203,144 / 3,278,576 / 3,203,788           3,203,788      2.35%
vit_v1_endpoint   1,121,580 / 1,167,652 / 1,152,440           1,152,440      4.11%
                                                              factor 2.780x
```

Before instrument subtraction. BERT line at the time: 3,575,304 -> 881,272 kB, 4.057x.

## Phase B evidence

The endpoint contains `s5_per_layer` (phase B); `v0` and `v1` gate values differ by design.

```
vit_v0_default    TRANSCRIPT|party=0|bytes=19769912480|rounds=1468   (all three replicates)
vit_v1_endpoint   TRANSCRIPT|party=0|bytes=19769912480|rounds=1468   (all three replicates)
```

- Byte- and round-identical in all six runs; no `transcript_artefact` needed.
- BERT: 11,228,688,384 bytes over 1,495 rounds. ViT: +76% bytes, 27 fewer rounds (197 tokens, no
  attention-mask chain).

`impl/equiv_per_layer_vit.py`, run inside the measured image:

```
ok  head order is layernorm -> CLS slice -> classifier, as the driver assumes
ok  12 layers, seq 197: decomposition is bit-identical to the whole model (max |logit| 4.619388e+00)
info  patch grid gives 197 tokens (196 patches + 1 CLS)
PASS
```

- Separate from BERT's test: phase B evidence is model-specific.
- ViT has no attention mask (`ViTLayer.forward(hidden_states, head_mask=None, ...)`); the check
  covers the head order instead (layernorm before the CLS slice).
- `ViTModel.forward` applies `self.layernorm` itself; the probe takes the encoder output. A first
  version normalised twice (4.09e-01 mismatch) and was corrected.

## Scope

- 2.780x on ViT against 4.057x on BERT is a result.
- The equivalence is shown on plaintext; the transcript tests that the protocol did the same work.
