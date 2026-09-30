# p_m1depth: the depth collapse follows the image

Not a step: one layer, 90 GB, `express`, `m1_minimal_att_keys`'s image (first ordering), read during
key generation and killed.

`m1_minimal_att_keys`'s table: 91.7% in `Ciphertext::operator=` (58.1%) and `Ciphertext::resize`
(33.6%). `m0_default` (same machine type, depth, recorder) resolves 134 sites.

```
machine    effective depth   image                      rank-1 caller
zen4              6          m0_default                 encrypt_zero_symmetric
express           6          m0_default                 encrypt_zero_symmetric
zen4              6          m1_minimal_att_keys        Ciphertext::operator= / resize
express           6          m1_minimal_att_keys        Ciphertext::resize          <- THIS RUN
bigsmp            6          m1_rtn_release_early       encrypt_zero_symmetric
```

- Re-deriving the published `m1` run with current tools reproduces its table to 0.1% (job 46380257).
- `patch_minimal_att_keys.py` replaces `keygen.create_galois_keys(gal_keys)` with
  `create_galois_keys(gal_steps_vector, gal_keys)`: another overload, another call chain; a fixed depth
  lands on another frame (README.md failure mode "a lever can change a row's name without moving one
  byte").
- `SNNI_PM_DEPTH_MAX` raisable since 2026-08-22; per-image depths would make the line's tables
  incomparable. Resolved by the chain recorder with `--program-root`.
