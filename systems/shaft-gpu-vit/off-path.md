# SHAFT-GPU-ViT: off-path

## Status

| row | VRAM (kB) | `W_vram` | host (kB) | `W_host` | cost | gate |
|---|---:|---:|---:|---:|---|---|
| `vit_v0_default` | 1,968,128 | 24.71 | 3,086,648 | 33.01 | baseline | |
| `vit_v1_endpoint` | 628,736 | 7.89 | 1,715,172 | 15.81 | paid (+18.8%) | equivalent |

| pool | BERT (`shaft-gpu`) | ViT (here) |
|---|---:|---:|
| VRAM | 18.40x | 3.13x |
| host | 2.05x | 1.80x |

- The device reduction on BERT removes the vocabulary term: the table share, `28996 * 768 * 8 B =
  173,976 kB`, device-resident for the one-hot lookup. ViT's embedding is a `768 x 3 x 16 x 16`
  patch projection (4,608 kB), so the ViT baseline starts at 1.97 GB (BERT 6.97 GB) and that term is
  absent.
- The host reduction comes from the per-layer driver, which removes the whole-model ONNX
  conversion; ViT has one too.

## Phase B

```
vit_v0_default    TRANSCRIPT|party=0|bytes=19769912480|rounds=1468   all three replicates
vit_v1_endpoint   TRANSCRIPT|party=0|bytes=19769912480|rounds=1468   all three replicates
```

- Byte- and round-identical. SHAFT-GPU's 524,288-byte artefact comes from the attention-mask chain,
  which ViT does not have.
- `steps/vit_v1_endpoint/impl/equiv_per_layer_vit.py`, run in this image:

```
ok  head order is layernorm -> CLS slice -> classifier, as the driver assumes
ok  12 layers, seq 197: decomposition is bit-identical to the whole model
PASS
```

## Defects found while building

| defect | fix |
|---|---|
| `D="$RESULTS/n12_seq128"` hard-coded three times in `shaft-gpu`'s sbatch; the first ViT run (results in `n12_seq197`) published `vram_reserved_peak_kb=NA` | result directory found by glob (`SHAFT_MAX_LENGTH` travels inside `SHAFT_INNER_ENV`, so a shell expansion would still give 128) |
| endpoint driver printed `subgraphs=15` for 14 (BERT's `n_layers + 3`) | count accumulated and checked against `n_layers + 2`; measured values unaffected |
| first six replicates ran with `ATTRIB_HOST=off ATTRIB_DEVICE=off` (direct `sbatch` bypassed `CHANNELS`) | discarded, re-run with the line's channel |

## Trades

None.
