# m14r_layernorm_release: measured one position early

Job 46884324 (run `m14r_tiled_ffn`; the build `m14r_layernorm_release` has the same binary, md5
668af2d5: set {tiled_ffn, rtn_release_ln1, rtn2_release_ln2, layernorm_release}), 2026-09-24/25,
`zen4`, 16-core cpuset, shared node, two layers, T=16, rc=0, wall 120,859 s, gate dc043a0b.
`levers.env`, `logs/` beside this file.

```
m13_rtn2_release_ln2   97,915,008 kB   93.38 GiB   (peak: Layer-1 attention)
m14r_layernorm_release 97,977,016 kB   93.44 GiB   +0.06%   (peak: Layer-1 attention)
```

- Lever fired (2 `LEVER|layernorm_release`); Layer-0 layernorm1 window 90.8 -> 82.6 GiB; global peak
  in the Layer-1 attention (93.4 GiB), where the copy is not resident.
- At m13's peak the largest object with a free fix is `enc_X_v`'s 12.1 GB capacity slack: position 14
  is `m14_enc_X_v_shrink`; layernorm_release is position 15 (`patch_layernorm_release.py` on the m14s
  image gives md5 e32c8a4b, identical to the m15r binary).
- Launched on the prediction that layernorm1 (90.9 in m10r) would outrank the Layer-1 attention (93.4
  in m11r); m13r's per-phase maxima: 2.9% apart (spread 2.0 pp).
