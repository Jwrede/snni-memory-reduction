# after_target_order3: device levers on top of `s3`

2026-09-29, n=3 each, `freeze_mode=maxparty`, recorder c2-32b38f744, token IDs from the cased
vocabulary. Not steps: after `s3` rule 3 selects the host (`W_host` 5.80 against `W_vram` 1.74),
which has no admissible target (`off-path.md`).

| run | configuration | VRAM peak | host peak | corrected wall (median) |
|---|---|---:|---:|---:|
| `s3_embed_chunk` (line) | | 378,880 kB | 1,680,764 kB | |
| `s4_param_spill_graphclear` | s3 + parameter spill + graph clear | 372,736 kB | 2,169,868 kB | 40.0 s |
| `s5_param_defer` | s3 + parameter spill + parameter defer | 362,496 kB | 2,167,132 kB | 40.0 s |
| `x_defer_nospill` | s3 + parameter defer, no spill | 366,592 kB | ~1.69 GB (marker) | ~43 s raw |

- Device peak at most -4.3%.
- The spill raises the host peak by 29%: spilled shares are read back through host buffers.
- Step directories are kept unchanged in this folder.
