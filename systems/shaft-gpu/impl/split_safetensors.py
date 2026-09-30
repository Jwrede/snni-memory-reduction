#!/usr/bin/env python3
"""Split a HF model.safetensors into per-scope shards (bert.embeddings, bert.encoder.layer.<i>,
bert.pooler, classifier), so the loader mmaps only the scope it needs and file-backed pages can't
accumulate in VmRSS across the 12 layers. Idempotent: skips if all expected shards exist."""

import os
import sys
from safetensors import safe_open
from safetensors.torch import save_file


def split(model_dir, n_layers=12):
    src = os.path.join(model_dir, "model.safetensors")
    out_dir = os.path.join(model_dir, "shards")
    os.makedirs(out_dir, exist_ok=True)

    expected = (
        ["bert.embeddings", "bert.pooler", "classifier"]
        + [f"bert.encoder.layer.{i}" for i in range(n_layers)]
    )
    if all(os.path.exists(os.path.join(out_dir, f"{s}.safetensors")) for s in expected):
        print(f"shards already present in {out_dir}")
        return out_dir

    with safe_open(src, framework="pt") as sf:
        keys = list(sf.keys())
        for scope in expected:
            shard = {}
            prefix = scope + "."
            for k in keys:
                if k.startswith(prefix):
                    shard[k[len(prefix):]] = sf.get_tensor(k)
            out = os.path.join(out_dir, f"{scope}.safetensors")
            save_file(shard, out)
            print(f"  wrote {scope}: {len(shard)} tensors -> {out}")
            del shard
    return out_dir


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/HC_Volume_105187419/moai/hf_cache/hub/models--andeskyl--bert-base-cased-sst2/snapshots/65343572032e352d48573d37e23c03609689696c"
    n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    out = split(model_dir, n_layers)
    print(f"shards directory: {out}")
