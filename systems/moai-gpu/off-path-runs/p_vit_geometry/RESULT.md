# p_vit_geometry: ViT-Base/16 packing

Run after the line closed; both configurations rc=0, gates byte-identical, one replicate each.

```
                     BERT   256 x 128        ViT   128 x 256      delta
s0_default           136,118,272 kB          136,740,864 kB       +0.46%
s13 endpoint          69,173,248              69,173,248           0 kB
factor                    1.9678x                 1.9768x
```

- 197 tokens padded to 256 and batch halved (the earlier campaign's recipe): `num_X 256 x num_row 128`
  and `num_X 128 x num_row 256`, both 32,768 slots. Same packed volume: not a sequence-length
  experiment. The endpoint peak depends on packed volume, not on the rectangle.
- Retires the earlier campaign's "BERT 72.3 GiB against ViT 73.8 GiB" (same padding and halving) as
  evidence of length insensitivity (MANIFEST corrected 2026-08-20).

## Observable added first

- `num_row` is a source constant; the sequence length is also a literal in `softmax.cuh:117`
  (`softmax`) and `softmax.cuh:486` (`softmax_boot`), on the measured path.
- A wrong-stride softmax walks the same modulus chain (the T1s gate passes).
- `impl/patch_vit_geometry.py` parameterises the constant and both literals (defaults = BERT values)
  and prints, with an abort if `num_batch * num_row != slot_count` or `num_batch != num_X`:

```
SNNI_GEOM|slot_count=32768|num_row=256|num_batch=128|num_X=128|at=softmax_boot
```
