# p_seq197: sequence length 197 (and the sweep)

Runs 46300785-46300792, 2026-08-19. Both configurations died after ~20 s on the device
(`cudaErrorMemoryAllocation`). Logs: `/scratch/tmp/j_wred02/sigma_gpu` (without `snni_campaign/`).

| config | error site | reserved |
|---|---|---|
| `s0_default` | `gpu_mem.cu:62` | 42,983,227,392 B (the 40 GiB warmup) |
| `s6_dealer_window` | `gpu_mem.cu:68` (same `cudaMallocAsync` in `gpuMalloc`, lines shifted by s6's patch) | 33,554,432 B |

Both after `FILL|nodes=97|weight_tensors=48|weight_elems=56641536|input_elems=151296` (197 x 768).
The harness then reports `FAILED: no transcript invariants, so this run has no gate observable`.

Refuted explanation: "62 GiB do not fit the job's card". One H200 (~143 GB) per gres unit
(`scontrol show node r18n01` -> `Gres=gpu:h200:8`); `gpu_mem.cu:48` sets
`cudaMemPoolAttrReleaseThreshold = UINT64_MAX`, `:50` the 40 GiB warmup.

## Sweep (endpoint image, one replicate each)

```
n_seq=128   rc=0   peak 2,139,204 kB   (0.02% from this line's published s6 figure)
n_seq=192   rc=1   gpu_mem.cu   cudaMallocAsync   code 2, out of memory
n_seq=196   rc=1   gpu_mem.cu   cudaMallocAsync   code 2, out of memory
n_seq=197   rc=1   gpu_mem.cu   cudaMallocAsync   code 2, out of memory
n_seq=256   rc=1   gpu_mem.cu   cudaMemcpy        code 1, invalid argument
```

- Not 64-divisibility (192 fails), not powers of two (256 fails). The power-of-two reading came from a
  note about MOAI-CPU's harness and was not tested here before.
- Resolved in `../p_seq256_vit/`: the key buffer `keyBufSz = 20 * OneGB` is a model-size constant;
  256 runs once it is raised. 197 still fails for a second reason.
- Related constants tuned to 128 tokens: SIGMA's key window (`op_high=825,125,000`), BumbleBee `b8`'s
  OT instance count, MOAI-GPU's softmax batch.
