# s7_keygen_tight: peak valid, table not at the peak

Sizes the dealer's three pinned host buffers (`patches/patch_keygen_tight.py`); declared from
`s6_dealer_window`'s table (`not_selectable_until: s6_dealer_window`, `step/levers.env`). Measured
twice on 2026-09-29, the second time with `freeze_mode=maxparty` and recorder c2-32b38f744, one run.

| run | host VmHWM (recorder table subtracted) | frozen host snapshot | gap |
|---|---:|---:|---:|
| 2026-09-29 (maxparty, c2) | 1,941,788 kB | 1,115,696 kB | -42.5 % |

- Peak valid: 1,941,788 kB (W_host 8.25 against 9.17 at `s6`), run time -13.4%.
- The host maximum is a short spike in key generation that has fallen before the freeze; the table is
  42.5% below the peak (limit 10%) on both measurements. No next step
  (`steps.pending/s8_weights_dead`) can be chosen from it.
- The line ends at `s6_dealer_window`. Step directory unchanged under `step/`.
