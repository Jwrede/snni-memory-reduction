"""s7 lever: loop CUDALongTensor's fp64 limb cross-products one at a time instead of materializing
the batched [nb^2,...] matmul (y_enc_span=[16,768,3072] fp64 = 302 MB for the FFN weight, since
cuBLAS has no int64 matmul). Result is BYTE-IDENTICAL to stock (same products, index/shift/sum),
memory only. Runtime monkeypatch, no CrypTen rebuild. Controlled by SHAFT_LIMB_LOOP (default off)."""
import os
import sys

import torch
import crypten.cuda.cuda_tensor as _ct

CUDALongTensor = _ct.CUDALongTensor

_ENABLE = os.environ.get("SHAFT_LIMB_LOOP", "0") == "1"

_nb = CUDALongTensor._CUDALongTensor__N_BLOCKS
_bks = CUDALongTensor._CUDALongTensor__BLOCK_SIZE
_bits = CUDALongTensor._CUDALongTensor__BITS
_encode = CUDALongTensor._CUDALongTensor__encode_as_fp64


def _limb_fp64(t, k):
    """fp64 tensor of limb k of the int64 tensor t (same expression __encode_as_fp64 uses per limb)."""
    v = (t >> (_bks * k)) & ((1 << _bks) - 1)
    d = v.double()
    return d.data if hasattr(d, "data") else d


def _limb_loop_matmul(x, y, *args, **kwargs):
    remove_x = remove_y = False
    if x.dim() == 1:
        x = x.view(1, x.shape[0])
        remove_x = True
    if y.dim() == 1:
        y = y.view(y.shape[0], 1)
        remove_y = True

    # Lazy limb encoding: one fp64 limb at a time (~19 MB per limb for the FFN weight, vs 4x19=75 MB).
    acc = None
    for i in range(_nb):
        y_i = _limb_fp64(y, i)
        for j in range(_nb):
            if (i + j) * _bks >= _bits:
                continue
            x_j = _limb_fp64(x, j)
            prod = torch.matmul(x_j, y_i, *args, **kwargs)
            if remove_x:
                prod = prod.squeeze(-2)
            if remove_y:
                prod = prod.squeeze(-1)
            term = prod.long() << ((i + j) * _bks)
            acc = term if acc is None else acc + term
            del x_j, prod, term
        del y_i
    return CUDALongTensor(acc)


if _ENABLE:
    _ct.HANDLED_FUNCTIONS[torch.matmul] = _limb_loop_matmul
    CUDALongTensor.matmul = staticmethod(_limb_loop_matmul)
    print(f"matmul_limb_loop_gpu: patched CUDALongTensor matmul "
          f"(nb={_nb}, bks={_bks})", file=sys.stderr, flush=True)
