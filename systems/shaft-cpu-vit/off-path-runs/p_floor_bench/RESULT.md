# p_floor_bench: ViT FFN weight-share size

Checks the weight-share term of the `shaft-cpu-vit` target: resident size of one secret-shared
`3072 x 768` FFN weight (`intermediate_size * hidden_size`) under CrypTen.

| | |
|---|---|
| job | 46762782, `express`, r05n10, 93 GB |
| image | `shaft-cpu-s0_default.sif` (torch 2.0.1, crypten 1.0.0) |
| wall | 5 s, rc 0 |
| files | `vit_floor_bench.py`, `vit_floor.sbatch`, `vit_floor-46762782.out` |

```
SHARE_DTYPE   torch.int64
SHARE_SHAPE   (3072, 768)
SHARE_BYTES   18874368 B = 18432 KiB   (= 3072 * 768 * 8)
RSS_DELTA     22472 KiB   (share isolated; ~4 MB CrypTen wrapper and torch caching allocator)
```

The share is a plain int64 tensor of the target shape, the largest tensor of the 81.5 MB target
(MANIFEST.md).
