# s2_plaintext_free: not a step

Jobs 45523104-15, 3 replicates, MIG slice, earlier ordering, gate byte-identical.

| | |
|---|---|
| selection | at s1's peak `W_vram` 10.01 < `W_host` 10.83; host rank 1 `anon:[heap]` (1576.6 MB, not targetable), `file:` is overhead, so `heap:libc10.so+0x605e5` (433.0 MB) |
| lever | `LEVER|plaintext_free_gpu|released_plaintext_bytes=433255432`, one fp32 BERT-base, after `from_onnx` |
| host peak | 2948356 -> 2949828 kB, +0.05% (spread 0.441%) |
| in-script marker peak | 3370368 -> 2131592 kB |
| row note | `host peak 2949828 kB is the kernel's VmHWM; the 100 ms poller only observed 2412348 kB (18.2% low), so this peak is a spike` |

Wall on the MIG slice (not controlled, GPU shared):

| step | levers | wall (median, corrected) | runtime_delta_pct |
|---|---|---:|---:|
| s0_default | none | 63 s | -- |
| s1_limb_loop | limb loop | 93.8 s | +74.8% |
| s2_plaintext_free | limb loop + release | 53.0 s | -43.5% |

Consequence: the system moved to a whole H200 (`MANIFEST.md`, Hardware).
