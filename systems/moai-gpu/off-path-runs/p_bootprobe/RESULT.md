# p_bootprobe: bootstrap working set

Job 46148052, `gpuh200`. Probe: one `snni_mem_marker` every 64 ciphertexts in the first bootstrap loop;
no allocation, arithmetic or ordering changed. On `s2_boot_input_release`'s tree.

Question: is the 768 x 34 MiB residual per bootstrap (`../../level-audit.md`) a retained per-call
workspace or the pool not reusing freed blocks?

```
after_selfoutput  58,930.1
boot1_ct0         58,950.1
boot1_ct64        60,230.1     +1,280.0
boot1_ct128       61,510.1     +1,280.0
boot1_ct192       62,790.1     +1,280.0
...                            +1,280.0 every 64, without deviation
boot1_ct704       73,030.1
after_bootstrap1  74,290.1
```

- 20.0 MiB per bootstrap = 21 MiB output at depth 20 minus 1 MiB input at depth 0 (released by `s2`).
  768 x 20 = 15,360.0; 58,930.1 + 15,360.0 = 74,290.1. No retained workspace.
- Earlier phases reproduce the audit (`keys_created` 40,498.1, `inputs_prepared` 68,146.1).

```
audit, tree without the release   58,930.1 -> 101,170.0   = +42,239.9
this probe, s2's tree             58,930.1 ->  74,290.1   = +15,360.0
                                                            -26,879.9
```

- `s2_boot_input_release` frees 768 MiB; the pool ends 26,879.9 MiB lower: released blocks are reused
  by the bootstrap's internal allocations. The audit's "workspace" was fragmentation.
- Remaining candidates in this phase are exhausted (`enc_ecd_x_copy` mod-switch is a no-op; the
  bootstrap is already per ciphertext).
- Policy knobs (release threshold, trim) fail on this allocator; shortening an object's lifetime inside
  a loop works.
