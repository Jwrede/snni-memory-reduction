"""Private embedding one-hot matmul split along the vocabulary (one chunk's Beaver triple resident).
Lookup over a secret index = matmul against a one-hot row over V=28996, drawing a full (V,D) triple;
triple + table + one-hot live together (950.2 MB, 41.5% of 2,289,520 kB; p_census_owner_s4). Split into
N=16 chunks (donor value), partial products summed. Port of SHAFT-GPU embed_chunk_gpu.py
(s2_embed_chunk). Phase B (transcript + equiv_embed_chunk). CPU port: no empty_cache(); malloc_trim(0)
kept (without it the host peak varies ~224 MB).
"""

import ctypes
import gc
import os
import sys

import torch

import crypten
import crypten.nn.module as _ctmod

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def _trim():
    """Return freed pages to the kernel instead of leaving them in glibc's arena."""
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


_original_embedding_forward = _ctmod.Embedding.forward


def _chunked_embedding_forward(self, input):
    weight, indices, _ = input
    voc_size = weight.shape[0]
    num_chunks = int(os.environ.get("SHAFT_EMBED_CHUNKS", "1"))

    # N=1 is the stock path verbatim, not a one-chunk emulation (which would draw masks in a
    # different order, so the no-op case would not be a no-op).
    if not crypten.is_encrypted_tensor(indices) or num_chunks <= 1:
        return _original_embedding_forward(self, input)

    provider = crypten.mpc.get_default_provider()
    r, v = provider.generate_one_hot_pair(indices.size(), voc_size, device=indices.device)
    j = indices - r
    j = j.get_plain_text().int().unsqueeze(-1) % voc_size
    arange1 = torch.arange(voc_size, device=indices.device).view([1, 1, voc_size]).repeat(
        (indices.shape[0], indices.shape[1], 1)
    )
    arange2 = (arange1 - j) % voc_size
    indices_oh = v.gather(2, arange2)
    del r, v, j, arange1, arange2
    _trim()

    chunk_size = (voc_size + num_chunks - 1) // num_chunks
    output = None
    for i in range(num_chunks):
        s = i * chunk_size
        e = min(s + chunk_size, voc_size)
        chunk_idx = indices_oh[..., s:e]
        chunk_w = weight[s:e, :]
        chunk_out = chunk_idx.matmul(chunk_w)
        del chunk_idx, chunk_w
        if output is None:
            output = chunk_out
        else:
            output = output + chunk_out
            del chunk_out
        _trim()

    del indices_oh
    _trim()
    return output


_ctmod.Embedding.forward = _chunked_embedding_forward

print(f"LEVER|embed_chunk_cpu|patched|chunks={os.environ.get('SHAFT_EMBED_CHUNKS', '1')}",
      file=sys.stderr, flush=True)
