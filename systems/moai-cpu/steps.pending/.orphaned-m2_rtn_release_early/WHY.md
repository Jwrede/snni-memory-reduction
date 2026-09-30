# m2_rtn_release_early: orphaned

- Derived from `m1_pool_new`'s peak (307,243,920 kB, `pm_depth=5`; rank 1 `seal::Ciphertext::resize`,
  51.5%); `m1_pool_new` was never on the line (then `m0_default -> m1_minimal_att_keys`, peak
  402,432,036 kB).
- Mechanism kept as calibration: releasing `rtn` after its last reader measured -3.88% on `bigsmp`
  (`off-path-runs/m1_rtn_release_early_bigsmp_d7/`).
- A revival needs re-derivation against its actual predecessor's table.
