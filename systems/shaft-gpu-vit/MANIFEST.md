# SHAFT-GPU-ViT: manifest

ViT transfer of SHAFT-GPU's line: the BERT line's levers applied unchanged to another model, on the
device and host pools.

| | |
|---|---|
| paradigm | pure MPC, CrypTen as vendored by SHAFT, 2 parties sharing one GPU |
| model | `google/vit-base-patch16-224`, bound at `/vit`, 12 layers, 197 tokens (196 patches + CLS) |
| cluster | PALMA, partition `gpuh200` |
| device | one whole NVIDIA H200, 141 GB, `--gres=gpu:1` |
| threads | `OMP_NUM_THREADS=8`, pinned |
| image | `shaft-gpu-estream.sif` (the BERT line's image) |
| drivers | `../shaft-gpu/impl/vit_instrumented_gpu.py` (baseline), `../shaft-gpu/impl/vit_perlayer_gpu.py` (endpoint) |
| pools | VRAM and host |
| published peak | median over runs of the maximum over parties |
| channel | `CHANNELS="-:PMREC=1,ATTRIB=1,SNNI_COMBINED=1"` |
| rows | `vit_v0_default` (baseline), `vit_v1_endpoint` (`SHAFT_LIMB_LOOP=1`, `SHAFT_EMBED_CHUNKS=16`, per-layer driver) |
| replicates | 3 |

## Build

```
impl_base_image: the image SHAFT-GPU's own steps run; see systems/shaft-gpu/MANIFEST.md
impl_artifacts_from: shaft-gpu
impl_build_docker: docker build -f ../shaft-gpu/impl/Dockerfile.gpu ../shaft-gpu
impl_build_palma: the BERT line's image, with the step env under ../shaft-gpu/impl/stepenv/
```

## Running

- No conf, `impl/` or sbatch of its own. Deploy and run `shaft-gpu` with the step ids
  `vit_v0_default`, `vit_v1_endpoint` (same ids here).

## Target

```
target_vram_kb: 79638
target_host_kb: 79638
overhead_vram_kb: 0
overhead_host_kb: measured_per_run
replicates: 3
gate_tier: T2
gate_tolerance: 1e-6
seed: crypten.manual_seed(0xDEADBEEF+rank, 0xC0FFEE+rank, 0x5EED) + torch.manual_seed(0x5EED)
gate_observable: the decrypted output logits (1000 ImageNet classes), `GATE|logits|...`
```

Same target as SHAFT-CPU-ViT (derivation: `systems/shaft-cpu-vit/MANIFEST.md`), carried by both
pools:

```
one layer's weights  = (4 * 768 * 768 + 2 * 768 * 3072) * 8     = 56623104 B
heaviest triple      = (197 * 768 + 768 * 3072 + 197 * 3072) * 8 = 24926208 B
target               =                                           81549312 B = 79638 kB
```

- `overhead_vram_kb` 0: the device peak is torch's caching-allocator high-water, which excludes the
  CUDA context.
- BERT target 217967 kB is 2.74x this one; `W` is not comparable to `shaft-gpu`'s.

## Driver differences from BERT

| | |
|---|---|
| attention mask | none (`ViTLayer` takes `head_mask=None`); 14 sub-graphs instead of 15; count checked against `n_layers + 2` with a hard stop |
| head | final LayerNorm, classifier on the CLS token (`ViTForImageClassification.forward`); no pooler |
| input | one `pixel_values` operand |
| checkpoint | local directory (image holds only BERT weights; nodes are offline) |
| geometry check | tokens = `(image_size / patch_size)^2 + 1`; the driver exits non-zero if `SHAFT_MAX_LENGTH` disagrees |

## Cross-check

```
shaft-cpu-vit  v0   TRANSCRIPT|party=0|bytes=19769912480|rounds=1468
shaft-gpu-vit  v0   TRANSCRIPT|party=0|bytes=19769912480|rounds=1468
```

Two separately written drivers on two machines, identical transcript.
