"""Embedding table never whole on the heap: each chunk's rows read from the spill file.
At x4_stream_load's peak (p_census_peak_x4) 445 MB of ~1,053 MB = one embedding table in three copies;
in use at the peak (graph_clear_values_cpu third null, +0.10% vs 0.03% spread). Composes with
encrypt_spill_cpu: empties the received weight's share (178 MB freed via a fresh _rewrap wrapper) and
reads each chunk's rows from the spill file (contiguous (V,D) int64 range:
np.fromfile(count=(e-s)*D, offset=s*D*8)). Donor: SHAFT-GPU embed_chunk_gpu SHAFT_EMBED_STREAM. Spill
record matched by shape; refuses unless exactly one matches. Free (same bytes, file, order); chunking
and its 16 mask draws are embed_chunk_cpu's; gate expected identical to x4's.
"""

import ctypes
import gc
import os
import sys

import numpy as np
import torch

import crypten
import crypten.nn.module as _ctm
from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None

_STATS = {"chunks": 0, "bytes": 0, "freed": 0}


def _trim():
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


def _spill_record_for(shape):
    """The spill record whose tensor has this shape, or a hard stop. Shape is enough only because it
    is CHECKED unique (the embedding is the one parameter of its size)."""
    spill = sys.modules.get("encrypt_spill_cpu")
    if spill is None:
        raise RuntimeError("embed_stream_cpu: encrypt_spill_cpu is not loaded; this lever reads "
                           "the files that one writes and cannot work without it")
    hits = [r for r in spill._SPILLED.values() if tuple(r[1]) == tuple(shape)]
    if len(hits) != 1:
        raise RuntimeError(f"embed_stream_cpu: {len(hits)} spill records have shape {tuple(shape)}, "
                           f"expected exactly 1; refusing to guess which file holds the embedding")
    return hits[0]


def _empty_in_place(x):
    """Free the payload of an already-materialised share, keeping the wrapper the graph holds."""
    inner = x if isinstance(x, ArithmeticSharedTensor) else getattr(x, "_tensor", None)
    if isinstance(inner, ArithmeticSharedTensor) and inner.share.numel():
        _STATS["freed"] += inner.share.numel() * inner.share.element_size()
        inner.share = torch.empty(0, dtype=torch.int64)


def _chunk_from_disk(path, scale, template, rows, cols, s, e):
    """Rows [s, e) of the spilled table, read without faulting in the rest."""
    n = (e - s) * cols
    arr = np.fromfile(path, dtype=np.int64, count=n, offset=s * cols * 8)
    if arr.size != n:
        raise RuntimeError(f"embed_stream_cpu: read {arr.size} of {n} int64 from {path}")
    _STATS["bytes"] += arr.nbytes
    _STATS["chunks"] += 1
    inner = ArithmeticSharedTensor.from_shares(torch.from_numpy(arr).view(e - s, cols))
    inner.encoder._scale = scale
    if isinstance(template, ArithmeticSharedTensor):
        return inner
    cls = type(template)
    new = cls.__new__(cls)
    new.__dict__.update(template.__dict__)
    new._tensor = inner
    return new


_ORIG = _ctm.Embedding.forward


def _streaming_embedding_forward(self, input):
    weight, indices, _ = input
    num_chunks = int(os.environ.get("SHAFT_EMBED_CHUNKS", "1"))
    if not crypten.is_encrypted_tensor(indices) or num_chunks <= 1:
        return _ORIG(self, input)

    inner = weight if isinstance(weight, ArithmeticSharedTensor) else getattr(weight, "_tensor", None)
    shape = tuple(inner.share.shape)
    voc_size, cols = shape
    path, _shape, scale, template = _spill_record_for(shape)

    # The table leaves the heap before any chunk is read; else the streamed slices add to the full
    # copy rather than replacing it and the peak rises.
    _empty_in_place(weight)
    _trim()

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
        if s >= e:
            continue
        chunk_w = _chunk_from_disk(path, scale, template, voc_size, cols, s, e)
        chunk_out = indices_oh[..., s:e].matmul(chunk_w)
        del chunk_w
        output = chunk_out if output is None else output + chunk_out
        _trim()

    del indices_oh
    _trim()
    return output


_ctm.Embedding.forward = _streaming_embedding_forward

print("LEVER|embed_stream_cpu|patched", file=sys.stderr, flush=True)


def _report():
    print(f"LEVER|embed_stream_cpu|chunks={_STATS['chunks']}|read_bytes={_STATS['bytes']}|"
          f"freed_bytes={_STATS['freed']}", file=sys.stderr, flush=True)


import atexit  # noqa: E402
atexit.register(_report)
