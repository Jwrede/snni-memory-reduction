#!/usr/bin/env python3
"""Diagnostic probe: is the ~574 MiB FFN-matmul triple-generation transient CrypTen MPC overhead or
cuBLAS/CUDA workspace? Separates PLAIN (plaintext matmul = cuBLAS baseline), CRYPTEN_COLD (with
one-time init) and CRYPTEN_WARM (steady per-matmul MPC cost). Marker PROF|<step>|maxmem_MiB."""
import os
import sys
import torch
import crypten
import crypten as ct

DEV = "cuda"


def m():
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def a():
    return torch.cuda.memory_allocated() / 1024 / 1024


def main():
    ct.init()
    rank = ct.comm.get().get_rank()

    # 1. PLAIN plaintext torch matmul -> cuBLAS workspace baseline
    xt = torch.randn(1, 128, 768, device=DEV)
    wt = torch.randn(768, 3072, device=DEV)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = a()
    zt = torch.matmul(xt, wt)
    torch.cuda.synchronize()
    print(f"PROF|PLAIN_matmul|resident_before={base:.0f}|maxmem={m():.0f}", file=sys.stderr, flush=True)
    del xt, wt, zt
    torch.cuda.empty_cache()

    # crypten inputs (FFN up-projection shape)
    X = ct.cryptensor(torch.randn(1, 128, 768)).to(DEV)
    W = ct.cryptensor(torch.randn(768, 3072)).to(DEV)
    torch.cuda.synchronize()

    # 2. CRYPTEN cold
    torch.cuda.reset_peak_memory_stats()
    b2 = a()
    Z1 = X.matmul(W)
    torch.cuda.synchronize()
    print(f"PROF|CRYPTEN_COLD|resident_before={b2:.0f}|maxmem={m():.0f}", file=sys.stderr, flush=True)

    # 3. CRYPTEN warm (init amortized)
    torch.cuda.reset_peak_memory_stats()
    b3 = a()
    Z2 = X.matmul(W)
    torch.cuda.synchronize()
    print(f"PROF|CRYPTEN_WARM|resident_before={b3:.0f}|maxmem={m():.0f}", file=sys.stderr, flush=True)

    if rank == 0:
        print(f"RESULT|rank0|plain_and_crypten_measured|final_resident={a():.0f}", flush=True)


if __name__ == "__main__":
    from multiprocess_launcher import MultiProcessLauncher
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()
