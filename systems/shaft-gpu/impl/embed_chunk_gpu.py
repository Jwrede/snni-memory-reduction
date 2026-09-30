"""Embedding-chunk lever (step s1), ported from CPU 07_embedding_chunked. Splits the (V,D) embedding
matmul (the VRAM peak, s0 ~5.7 GB) along V into SHAFT_EMBED_CHUNKS chunks, so only one chunk's Beaver
triple is resident at a time. torch.cuda.empty_cache() (device analog of malloc_trim) is part of the
lever, needed for the reserved wall to drop. Import after crypten; SHAFT_EMBED_CHUNKS=1 is a no-op."""
import os
import sys
import gc
import ctypes

import torch
import crypten
import crypten.nn.module as _ctmod

try:
    _libc = ctypes.CDLL("libc.so.6")
except Exception:
    _libc = None

_USE_CUDA = os.environ.get("SHAFT_DEVICE", "cuda") == "cuda" and torch.cuda.is_available()

# Table streaming (s13): keep the (V,d) table in host memory, move only the current chunk's slice to
# the device. Placement move: takes the 188 MB table off VRAM at the cost of H2D per chunk.
_STREAM_TABLE = os.environ.get("SHAFT_EMBED_STREAM", "0") == "1"
_DEV = os.environ.get("SHAFT_DEVICE", "cuda")


def _trim():
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass
    if _USE_CUDA:
        torch.cuda.empty_cache()


_original_embedding_forward = _ctmod.Embedding.forward


def _chunked_embedding_forward(self, input):
    weight, indices, _ = input
    voc_size = weight.shape[0]
    num_chunks = int(os.environ.get("SHAFT_EMBED_CHUNKS", "1"))

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
        if _STREAM_TABLE and _USE_CUDA:
            chunk_w = chunk_w.to(_DEV)      # table lives on the host; only this slice goes to VRAM
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
print(f"embed_chunk_gpu: Embedding.forward chunked "
      f"(chunks={os.environ.get('SHAFT_EMBED_CHUNKS', '1')}, cuda={_USE_CUDA})",
      file=sys.stderr, flush=True)
