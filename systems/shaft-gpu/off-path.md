# SHAFT-GPU: off-path

Measured changes outside the line `s0`-`s3`, and where the line ends. Run directories:
`off-path/<run>/` and `off-path-runs/<run>/`. Earlier orderings are archived outside the repository
under `/mnt/HC_Volume_105187419/offload/reorder_archive/shaft-gpu/`.

## Status

Measured 2026-09-29, n=3, `freeze_mode=maxparty`, recorder c2-32b38f744, token IDs from the cased
vocabulary.

| step | VRAM peak (kB) | `W_vram` | pool attacked | cost |
|---|---:|---:|---|---|
| `s0_default` | 6,971,392 | 31.98 | | baseline |
| `s1_limb_loop` | 2,406,400 | 11.04 | VRAM | free |
| `s2_per_layer` | 1,468,416 | 6.74 | host (rule 3) | paid |
| `s3_embed_chunk` | 378,880 | 1.74 | VRAM | paid; -94.6%, 18.400x |

- Both pools carry the same target (217,967 kB). After `s3` the host is at 1,680,764 kB
  (`W_host` 5.80), above the device (`W_vram` 1.74), so rule 3 selects the host, which has no
  admissible target (below). The line ends.
- `s2_per_layer` is phase B with a declared transcript artefact: the transcript differs from its
  predecessor's by exactly 524,288 bytes in one round, a removal, declared in advance; the same
  difference is in both SHAFT baselines with no lever, so it belongs to the ONNX export
  (`steps/s2_per_layer/RESULT.md`).

## End of the line: host pool at `s3`

| object at `s3`'s host peak (1,680,764 kB) | MB | frac | disposition |
|---|---:|---:|---|
| `anon:[heap]` | 494.6 | 0.287 | `anon:`, never a target |
| `file:db532512...db6f5` | 431.0 | 0.250 | model weights blob, page cache (below) |
| `heap:libcuda.so.1+0x2a13c2` | 95.0 | 0.055 | CUDA driver |
| `heap:libcuda.so.1+0x1350750` | 89.9 | 0.052 | CUDA driver |
| `file:(deleted)` | 86.2 | 0.050 | largest remaining program object |

### The model blob

smaps at the earlier `s5` endpoint (same row at `s3`):

```
14824859a000-1482622cd000 rw-p 00000000
  /root/.cache/huggingface/hub/models--andeskyl--bert-base-cased-sst2/blobs/db5325...
Size 423116 kB   Rss 422388   Pss 211958
Shared_Clean 420860   Private_Clean 1528   Private_Dirty 0   Anonymous 0   Swap 0
```

| property | consequence |
|---|---|
| `Private_Dirty 0`, `Anonymous 0` | clean page cache; the kernel can drop and re-read it |
| `Pss` = `Rss` / 2 | both parties map the same blob; `VmHWM` charges each the full size |
| available fix | `madvise(MADV_DONTNEED)` after `from_pretrained`: -422 MB of `VmHWM`, no run time |
| why not a step | it lowers the metric without lowering memory the program needs |
| overhead? | counted in the peak (decided 2026-08-21): no other system's peak carries a mapped model blob, so deducting it would be a per-system metric change |

At the `s5` endpoint: counted `W` 8.30 (overhead 418,936 kB), deducted `W` 6.36 (overhead
842,052 kB), peak 2,228,136 kB.

## Device levers measured after `s3`

`off-path-runs/after_target_order3/RESULT.md`, n=3 each, on top of `s3`:

| run | configuration | VRAM peak | host peak | corrected wall |
|---|---|---:|---:|---:|
| `s4_param_spill_graphclear` | s3 + parameter spill + graph clear | 372,736 kB | 2,169,868 kB | 40.0 s |
| `s5_param_defer` | s3 + parameter spill + parameter defer | 362,496 kB | 2,167,132 kB | 40.0 s |
| `x_defer_nospill` | s3 + parameter defer, no spill | 366,592 kB | ~1.69 GB (marker) | ~43 s raw |

At most -4.3% VRAM; the spill raises the host by 29% (shares read back through host buffers).
Rule 3 does not select the device pool while the host is further above the target.

## Host dispositions measured on the MIG slice

Before the move to the full card; the device pool reproduced byte for byte across both machines,
the host pool within 4.5%.

### Plaintext model copy (`off-path/s2_plaintext_free`)

| | |
|---|---|
| object | `heap:libc10.so+0x605e5`, 433.0 MB: `crypten/nn/onnx_converter.py:64` stores `copy.deepcopy(pytorch_model)`, read only by `Module.to_pytorch()` (never called) |
| jobs | 45523104-15, 3 replicates, gate byte-identical |
| lever | `LEVER|plaintext_free_gpu|released_plaintext_bytes=433255432` |
| in-script marker peak | 3,370,368 -> 2,131,592 kB |
| host peak | 2,948,356 -> 2,949,828 kB, +0.05% (spread 0.441%) |
| cause | 70.2% of the host high-water in `after_model_load -> after_model_convert` (+2,331,056 kB in 33 s); glibc keeps the freed conversion pages (`anon:[heap]` 1420.4 MB at s0, 984.0 MB at s1) |
| disposition | basis of `# pool_skipped: host` in that ordering |

### Encrypted parameters on the host (`off-path/s2_param_stream`)

| | |
|---|---|
| change | parameters kept on the host; each operator gets a device copy for its own call |
| VRAM | 2,406,400 -> 1,695,744 kB, -29.5% |
| host | +87,732 kB, +2.5% (8x the host spread) |
| verdict | placement (opposite signs), not a reduction |
| note | 865.8 MB moved to the host, host peak +87.7 MB: `anon:[heap]` already held 84% of those pages |

## Phase B and the 1-layer gate (`embed_chunk`)

| | |
|---|---|
| history | filed here 2026-08-02, moved into the line the same day as the first phase B step; pre-rule runs in `off-path/superseded-s2_embed_chunk-pretranscript/` |
| transcript | 11,229,212,672 bytes in 1526 rounds against the baseline's 1496 |
| 1-layer gate (jobs 45639682/83, `SHAFT_N_LAYERS=1`) | logit 0 moves 8.85e-4 absolute, 6.2e-3 relative |
| why not used | two unchanged 12-layer runs are byte-identical here, so the 12-layer gate works; BumbleBee moves its gate only because its 12-layer runs differ by 3.4x |

## Not trades, recorded for reference

| item | status |
|---|---|
| polynomial GELU with fewer softmax iterations | accuracy trade, not measured here; earlier campaign `[REJECTED]` on this system, 684 -> 684 MiB on SHAFT-CPU |
| VRAM-only line of 2026-07-26 (s0..s3) | chose steps on VRAM alone; host `W` was larger after s1 (18.2 against 10.5) and s2 (9.4 against 4.3); `s1_embed_chunk` VRAM -5161 MB, host +524 MB; replayed by `tools/test_conformance.sh`, which must reject it |
| `unattributed_vram` 0.2088 at `s0` | `reserved` minus `live`, caching-allocator overhead, ~1483 MB |
| pinned seeds | part of the measured system, applied to every step; a security defect in deployment |

---

# Earlier stages of this line

## `encrypt_chunk_gpu` refuted from markers (2026-08-18, on `s4_param_defer`)

SHAFT-CPU's `s8_encrypt_chunk` (-18.4% there) was ported and not run:

```
after_init            rss=  433,144   hwm=  433,144
after_model_load      rss=  608,292   hwm=  608,292
after_model_convert   rss=1,908,960   hwm=2,911,044   <- the entire high-water is created here
after_model_encrypt   rss=1,822,684   hwm=2,911,044
after_model_cuda      rss=1,912,428   hwm=2,911,044
after_input_encrypt   rss=1,912,696   hwm=2,911,044
... every graphop node through the whole inference: hwm=2,911,044, unchanged
```

- The host high-water is set inside `ct.nn.from_pytorch(...)`; `Module.encrypt()` runs later.
- On SHAFT-CPU the peak was inside `enc_embed.encrypt()` (601,612 kB transient) because its
  per-layer driver had already removed the conversion peak.
- The conversion leaves +1,300,668 kB resident and raises the high-water by +2,302,752 kB: ~1 GB
  transient inside `from_pytorch`. Host tables at that peak: `heap:python3.10+0xd6be1` 870.5 MB,
  `heap:libstdc++.so.6.0.29+0xf3269` 799.0 MB.
- Levers against it: `onnx_single_file_export_gpu` refuted
  (`off-path-runs/s3_onnx_single_file_export`); the per-layer driver removes the event (now
  `s2_per_layer`).
