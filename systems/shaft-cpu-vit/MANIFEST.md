# SHAFT-CPU-ViT: manifest

ViT transfer of SHAFT-CPU's line: the BERT line's lever stack applied unchanged to another model.
A separate system because a different model is a different operating point.

| | |
|---|---|
| paradigm | pure MPC, CrypTen as vendored by SHAFT, 2 parties on the host CPU |
| model | `google/vit-base-patch16-224`, bound at `/vit`, 12 layers, 197 tokens (196 patches + CLS) |
| cluster | PALMA, partition `express`, `--exclude=r07n04`, `--exclusive` |
| node | 36-core Skylake AVX-512 (2 x 18), 95 GB |
| threads | `OMP_NUM_THREADS=8`, pinned |
| image | `shaft-cpu-s0_default.sif`, sha256 `f35e71f6d2...` (the BERT line's image) |
| drivers | `impl/vit_instrumented_cpu.py` (baseline), `impl/vit_perlayer_stream_cpu.py` (endpoint), under `systems/shaft-cpu/` |
| pools | host |
| published peak | median over runs of the maximum over parties |
| rows | `v0_default` (baseline), `v1_endpoint` (`SHAFT_EMBED_CHUNKS=16`, `SHAFT_ENC_CHUNKS=8`, per-layer stream) |
| replicates | 3 |

## Build

```
impl_base_image: the image SHAFT-CPU's own steps run; see systems/shaft-cpu/MANIFEST.md
impl_artifacts_from: shaft-cpu
impl_build_docker: docker build -f ../shaft-cpu/impl/Dockerfile.cpu ../shaft-cpu
impl_build_palma: the BERT line's image, with the step env under ../shaft-cpu/impl/stepenv/
```

## Running

- No conf, `impl/` or sbatch of its own (one copy of the instrument settings, `tools/check_policy.sh`).
- Deploy and run `shaft-cpu` with the step ids `vit_v0_default`, `vit_v1_endpoint`; the `vit_`
  prefix selects the ViT branch and the checkpoint bind.
- Harvested here as `v0_default`, `v1_endpoint`; `RUN_META` records the cluster name.

## Target

```
target_host_kb: 79638
overhead_host_kb: measured_per_run
replicates: 3
gate_tier: T2
gate_tolerance: 1e-6
seed: crypten.manual_seed(0xDEADBEEF+rank, 0xC0FFEE+rank, 0x5EED) + torch.manual_seed(0x5EED)
gate_observable: the decrypted output logits, printed as `GATE|logits|...` by the driver
phase: B from the endpoint on, with both conditions measured -- see below
```

Rule as on SHAFT-CPU: the working set of the higher phase under per-layer loading. ViT has no
vocabulary (patch projection `768 * 3 * 16 * 16` values), so the layer phase is the higher: one
layer's weight shares plus the heaviest Beaver triple (FFN up-projection, `s = 197`), 8 B per share.

```
one layer's weights  = (4 * 768 * 768 + 2 * 768 * 3072) * 8     = 56623104 B
heaviest triple      = (197 * 768 + 768 * 3072 + 197 * 3072) * 8 = 24926208 B
target               =                                           81549312 B = 79638 kB = 81.5 MB
```

- Weight-share term checked by `off-path-runs/p_floor_bench/RESULT.md`: one `3072 x 768` FFN weight
  share is 18874368 B.
- BERT target 217967 kB is 2.74x this one; `W` is not comparable between the two systems, the
  reduction factor is.

## Gate: phase B

`v1_endpoint` contains the per-layer driver (phase B): converting and encrypting layer i+1 while
layer i's result exists interleaves PRZS and Beaver draws, so `v0` and `v1` logits differ.

```
v0_default    TRANSCRIPT|party=0|bytes=19769912480|rounds=1468   all three replicates
v1_endpoint   TRANSCRIPT|party=0|bytes=19769912480|rounds=1468   all three replicates
```

`../shaft-cpu/impl/equiv_per_layer_vit.py` in the measured image (output in
`steps/v1_endpoint/impl/equiv_per_layer_vit.out`):

```
ok  head order is layernorm -> CLS slice -> classifier, as the driver assumes
ok  12 layers, seq 197: decomposition is bit-identical to the whole model
PASS
```

Record of the test's first, miswired version: `../shaft-cpu/off-path-runs/vit_transfer/RESULT.md`.
