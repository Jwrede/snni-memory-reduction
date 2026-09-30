"""Encrypt a large parameter in slices, so its three simultaneous copies are one Nth each. SHAFT-GPU
port of shaft-cpu s8_encrypt_chunk. Targets the ~601,612 kB transient inside enc_embed.encrypt()
that no marker sees. Only the embedding (22.3M elements) exceeds the 5M threshold, so exactly one
parameter is chunked. First-category free; in phase B (broadcasts the mask stream in N pieces)."""

import ctypes
import gc
import os
import sys

import torch

import crypten
import crypten.nn.module as _ctm

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None

_STATS = {"chunked": 0, "chunks": 0, "elements": 0}

# Order check: param_spill_gpu captures Module.encrypt at its import time, so this lever (which
# REPLACES it) must load first, else the chunking is patched over and silently never runs.
if "param_spill_gpu" in sys.modules:
    print("FATAL|encrypt_chunk_gpu|param_spill_gpu is already loaded, so it has captured the "
          "unchunked Module.encrypt and this lever would never run. Put encrypt_chunk_gpu BEFORE "
          "param_spill_gpu in SHAFT_LEVERS.", file=sys.stderr, flush=True)
    raise RuntimeError("encrypt_chunk_gpu must load before param_spill_gpu")


def _trim():
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


def _chunked_cryptensor(plaintext, num_chunks, **kwargs):
    """The donor's body: encrypt slice by slice, releasing between slices, then concatenate."""
    splits = torch.chunk(plaintext, num_chunks, dim=0)
    enc_chunks = []
    for c in splits:
        # `.contiguous()` so the encryption never sees a view whose base is the whole parameter,
        # which would keep the plaintext alive for the entire loop.
        enc_chunks.append(crypten.cryptensor(c.contiguous(), **kwargs))
        _trim()
    out = crypten.cat(enc_chunks, dim=0)
    del enc_chunks
    _trim()
    return out


_ORIG_ENCRYPT = _ctm.Module.encrypt


def encrypt(self, mode=True, src=0):
    """`Module.encrypt` with only the parameter loop replaced (chunked above the threshold); the
    buffer walk and recursion are the stock body."""
    if mode == self.encrypted:
        return self

    threshold = int(os.environ.get("SHAFT_ENC_CHUNK_THRESHOLD", "5000000"))
    num_chunks = int(os.environ.get("SHAFT_ENC_CHUNKS", "1"))

    self.encrypted = mode
    for name, param in list(self.named_parameters(recurse=False)):
        requires_grad = param.requires_grad
        if mode:
            if num_chunks > 1 and param.numel() >= threshold:
                enc = _chunked_cryptensor(param, num_chunks, src=src,
                                          requires_grad=requires_grad)
                _STATS["chunked"] += 1
                _STATS["chunks"] += num_chunks
                _STATS["elements"] += param.numel()
            else:
                enc = crypten.cryptensor(param, src=src, requires_grad=requires_grad)
            self.set_parameter(name, enc)
        else:
            self.set_parameter(name, param.get_plain_text())
            self._parameters[name].requires_grad = requires_grad

    for name, buffer in self.named_buffers(recurse=False):
        if mode and torch.is_tensor(buffer):
            self.set_buffer(name, crypten.cryptensor(buffer, src=src, requires_grad=False))
        elif isinstance(buffer, crypten.CrypTensor):
            self.set_buffer(name, buffer.get_plain_text())
            self._buffers[name].requires_grad = False

    return self._apply(lambda m: m.encrypt(mode=mode, src=src))


_ctm.Module.encrypt = encrypt

# The announcement wording is load-bearing: check_lever in shaft_gpu_campaign.sbatch greps stderr
# for "<module>: patched" and exits 5 when a requested lever is silent.
print(f"encrypt_chunk_gpu: patched Module.encrypt "
      f"(chunks={os.environ.get('SHAFT_ENC_CHUNKS', '1')}, "
      f"threshold={os.environ.get('SHAFT_ENC_CHUNK_THRESHOLD', '5000000')})",
      file=sys.stderr, flush=True)


def _report():
    print(f"LEVER|encrypt_chunk_gpu|chunked_params={_STATS['chunked']}|"
          f"chunks={_STATS['chunks']}|elements={_STATS['elements']}",
          file=sys.stderr, flush=True)


import atexit  # noqa: E402
atexit.register(_report)
