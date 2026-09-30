# s3_onnx_single_file_export: refuted

SHAFT-CPU's ONNX lever ported; earlier ordering, base `s2_embed_chunk`. Off-path. Module
`impl/onnx_single_file_export_gpu.py`, declaration `impl/stepenv/s3_onnx_single_file_export.env`.

```
s2_embed_chunk (base)  2,942,724 / 2,948,232 / 2,944,620   median 2,940,520   spread 0.187%

export to GPFS         3,283,444 / 3,325,248 / 2,671,944   median 3,283,444   spread 19.9%   +11.7%
export to /tmp         3,330,304 / 3,184,228 / 2,677,672   median 3,184,228   spread 20.5%    +8.3%
```

VRAM 1,546,240 kB in all six runs.

```
LEVER|onnx_single_file_export_gpu|onnx_bytes=433533725|exports=1|sym_registry=False|dir=/results|fstype=gpfs
LEVER|onnx_single_file_export_gpu|onnx_bytes=433533725|exports=1|sym_registry=False|dir=/tmp|fstype=rootfs
```

## Selection

Rule 3: `W_vram` 8.89 below `W_host` 13.25 at s2, host selected. Rank at s2's host peak:

```
870.5 MB  28.9%  heap:python3.10+0xd6be1            CPython serving io.BytesIO
867.0 MB  28.8%  heap:libstdc++.so.6.0.29+0xf3269   the protobuf parse
433.3 MB  14.4%  file:...bert-base-cased-sst2/blobs  the checkpoint, mmap'd by torch
354.7 MB  11.8%  anon:[heap]                        located, never a target
```

870.5 MB = 2 x 433 MB payload (BytesIO doubling). The `file:` row is the checkpoint, not an export.

## Result

- Second triple on node-local `/tmp` (`fstype=rootfs`) rules out GPFS client buffers.
- Both triples are bimodal: modes ~570 MB apart (3.2-3.3 GB twice, ~2.68 GB once).
- Conversion-phase attacks across both systems:

| system | change | effect |
|---|---|---|
| shaft-cpu | release the plaintext model | peak unchanged, inference +2 to 4% |
| shaft-cpu | same with `malloc_trim(0)` | peak unchanged, inference +2 to 4% |
| shaft-cpu | release the protobuf initializers | peak unchanged, inference +2 to 4% |
| shaft-gpu | BytesIO replaced by a file | peak +8 to 12%, spread 20% |

The same lever is part of SHAFT-CPU's `s4_onnx_and_spill`; it does not transfer here.
