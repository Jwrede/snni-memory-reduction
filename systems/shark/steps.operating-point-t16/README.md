# SHARK at T=16, August study

Measured 2026-08-04, 18 runs (six steps, 3 replicates), same images, levers and policy 2026-07-29.1
as the T=4 line, `OMP_NUM_THREADS=16`, leader-only freeze. Superseded as the published line by the
re-measurement of 2026-09-30 (T=16, `freeze_mode=maxparty`, `MANIFEST.md`). Determinism re-verified
at T=16 first (job 45755748: 0 of 98,304 elements differ on either party over two unchanged runs).

| step | T=4 peak (kB) | T=16 peak (kB) | diff | T=4 cost | T=16 cost |
|---|---:|---:|---:|---|---|
| s0_default | 58,101,620 | 58,091,704 | -0.02% | -- | -- |
| s1_batch_check_stream | 56,894,512 | 56,895,676 | +0.00% | free | free |
| s2_layer_weights | 56,286,412 | 56,288,764 | +0.00% | free | free |
| s3_key_stream | 1,201,740 | 1,203,580 | +0.15% | free | paid (+14.1%) |
| s4_ars_charge | 513,412 | 514,704 | +0.25% | free | paid (+11.0%) |
| s5_up_split | 470,796 | 472,484 | +0.36% | free | free |

- Peaks: median over replicates of each run's maximum over the two parties, raw `VmHWM`. Within
  0.36% across a 4x change in threads.
- Corrected wall spreads: 2.4 to 4.2% at T=16, 2.6 to 9.6% at T=4. Key streaming +14.1% at 2.4%
  spread (T=16) against +6.9% at 7.5% (T=4): resolvable at one operating point only.
- In this order the T=16 line has a free step after two paid ones (rule 6); the published line
  admits `s5_up_split` after the paid steps (README.md).
- `s4_ars_charge` was re-measured with `SHARK_FREEZE_LEADER=1` after the derive refused the first
  attempt: at `s4` party 1 sits 8.4% above the freeze leader (7.2 to 9.0% at T=4).
