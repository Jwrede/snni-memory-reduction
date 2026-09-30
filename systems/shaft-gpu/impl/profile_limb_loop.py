#!/usr/bin/env python3
"""Correctness + memory check for the s7 limb-loop matmul patch (FFN up-projection shape, single
process). Computes the same int64 matmul with stock CUDALongTensor.matmul and _limb_loop_matmul,
checks they are identical, and reports peak max_memory_allocated for each."""
import sys
import torch
import crypten.cuda.cuda_tensor as _ct
import matmul_limb_loop_gpu as mll   # SHAFT_LIMB_LOOP unset -> defines _limb_loop_matmul, does NOT patch

CUDALongTensor = _ct.CUDALongTensor
DEV = "cuda"


def mib():
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def main():
    torch.manual_seed(0)
    xi = torch.randint(-(2 ** 40), 2 ** 40, (1, 128, 768), dtype=torch.long, device=DEV)
    wi = torch.randint(-(2 ** 40), 2 ** 40, (768, 3072), dtype=torch.long, device=DEV)
    X = CUDALongTensor(xi)
    W = CUDALongTensor(wi)

    # stock (dispatches to the registered CUDALongTensor matmul)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    Zs = torch.matmul(X, W)
    torch.cuda.synchronize()
    sp = mib()
    zs = Zs.tensor().clone()

    # limb-loop version
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    Zl = mll._limb_loop_matmul(X, W)
    torch.cuda.synchronize()
    lp = mib()
    zl = Zl.tensor().clone()

    diff = (zs - zl).abs().max().item()
    print(f"CHECK|maxabsdiff={diff}|stock_peak_MiB={sp:.0f}|loop_peak_MiB={lp:.0f}|"
          f"Zshape={tuple(zs.shape)}", flush=True)


if __name__ == "__main__":
    main()
