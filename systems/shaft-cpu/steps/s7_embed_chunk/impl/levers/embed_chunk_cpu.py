"""Split the private embedding's one-hot matmul along the vocabulary, so only one chunk's Beaver
triple is resident at a time.

THE OBJECT, from the owner census at s4's own PUBLISHED peak (`p_census_owner_s4`, rss 2,332,684 kB
against a 2,336,656 kB peak, so the snapshot is the peak to 0.17%):

    534.5 MB  n=3   torch.int64 (28996, 768)      the embedding table
    237.5 MB  n=8   torch.int64 (1, 128, 28996)   the one-hot intermediates
    178.2 MB  n=1   torch.int64 (22268928,)       28996 x 768 flat, a fourth table-sized block
    --------
    950.2 MB, i.e. 41.5% of the 2,289,520 kB peak, and every row of it is embedding-shaped.

The census deduplicates by storage `data_ptr`, so these are distinct allocations rather than views
counted twice.

WHAT CRYPTEN DOES, and why the peak sits here. A private embedding lookup over a secret index cannot
index; it has to be a matmul against a one-hot row over the WHOLE vocabulary V = 28996. So
`Embedding.forward` builds a (B, S, V) one-hot pair, reveals the masked index, gathers the row, and
multiplies it by the (V, D) table. That last matmul draws a Beaver triple of the full (V, D) shape,
and the triple, the table and the one-hot all have to be live at the same instant.

THE LEVER splits that matmul along V into N chunks and sums the partial products, so at most 1/N of
the triple exists at once. Steps 1 and 2 -- the one-hot pair and the reveal -- are untouched.

DONOR: SHAFT-GPU's `impl/embed_chunk_gpu.py` (its published `s2_embed_chunk`), which measured
2,406,400 -> 1,546,240 kB of VRAM there. A within-campaign donor on the SAME codebase and the same
CrypTen, so this is a port rather than a transfer. Its own docstring records that it was itself
ported from a CPU monkeypatch in the retired campaign, which is why the arithmetic below is the
GPU file's with the device handling removed.

N = 16 IS OUR PARAMETER, not the system's, and it is declared as such. It is the donor's value.

WHY IT IS NOT FIRST CATEGORY, stated rather than glossed. Splitting along V does not only make the
triple and the PRZS masks smaller, it makes them MORE NUMEROUS: each chunk draws its own. On a
fixed-point protocol the truncation error is share-dependent, so a different mask stream is a
different output, and the exact gate cannot tell that from a broken protocol. This is therefore a
PHASE B step and has to carry the two conditions SHAFT-GPU's row carries: the transcript (same bytes
exchanged, more envelopes) and a plaintext equivalence test of the chunked arithmetic against the
matmul it replaces.

WHY THE CPU PORT DROPS `torch.cuda.empty_cache()` AND KEEPS `malloc_trim`. On the GPU the caching
allocator holds freed blocks as RESERVED, which is that system's published metric, so emptying the
cache is part of the lever there. Here the metric is resident host memory and the analogous holder
is glibc's arena. `malloc_trim(0)` is what returns it, and this line has measured that it matters:
without it the host peak takes one of two values about 224 MB apart depending on whether the arena
happened to be trimmed (see `off-path-runs/s5_graph_clear_values/RESULT.md`).
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

    # N=1 is the stock path verbatim, not a one-chunk emulation of it: an emulation would draw its
    # masks in a different order and the no-op case would not be a no-op.
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
