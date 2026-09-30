# m1_prerecorder: m1_minimal_att_keys before the recording pool

Published as `m1_minimal_att_keys` until 2026-08-28; moved because the line changed instrument (the
whole line must carry the recording SEAL pool, MANIFEST.md).

```
published here    381,080,936 kB   wall 38,733 s   -5.31% against m0's 402,436,136
published now     381,470,412 kB   wall 38,676 s   -5.20% against m0's 402,401,204
                      +0.1022%          -0.15%
```

- `gate.txt` bit-identical over 7,680 values (md5 `712aece2473c7980b998851e3f931709`).
- Step effect -5.31% against -5.20% (0.11 pp).
- Old rank 1: `seal::Ciphertext::operator=(seal::Ciphertext const&) <- seal::util::MemoryPoolHeadMT::get()`,
  220,882.1 MB, `named_frac` 0.5797. The replacement names the same 46.9% as `sealpool:retained`,
  `named_frac` 1.0.
- `peak-objects.md` beside this file: the decomposition as it was published.
