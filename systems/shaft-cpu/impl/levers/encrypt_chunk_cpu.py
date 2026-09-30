"""Large parameters encrypted in slices (three simultaneous copies become 1/N each).
Basis: s7_embed_chunk's peak 1,080,140 kB vs highest phase marker 393,464 (2.7x): transient inside
enc_embed.encrypt() between embed|after_convert and embed|after_encrypt (601,612 kB; = anon:residual,
711.6 MB, 64.3% of the endpoint). One parameter's encryption holds plaintext 94 MB + int64 share 187 +
mask share 187 + outgoing share 187 = ~570 MB. Mechanism from retired 08_low_cpu_mem_load: parameters
above 5,000,000 elements (only the embedding, 22.3M; next 2.36M) split into N=8 along dim 0, encrypted,
crypten.cat, collect + malloc_trim between (~70 MB).
Load order required: replaces Module.encrypt, so it must load before encrypt_spill_cpu captures the
original (checked below). Free. Phase B expected: cryptensor(x, src=0) broadcasts a share; 8 slices =
8 broadcasts (same bytes, more rounds, mask stream in 8 pieces).
"""

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

# Order check: encrypt_spill_cpu captures Module.encrypt at its import; if already loaded, replacing
# the method now leaves the spill lever calling the stock one and this lever silently does nothing.
if "encrypt_spill_cpu" in sys.modules:
    print("FATAL|encrypt_chunk_cpu|encrypt_spill_cpu is already loaded, so it has captured the "
          "unchunked Module.encrypt and this lever would never run. Put encrypt_chunk_cpu BEFORE "
          "encrypt_spill_cpu in SHAFT_LEVERS.", file=sys.stderr, flush=True)
    raise RuntimeError("encrypt_chunk_cpu must load before encrypt_spill_cpu")


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
        # .contiguous() so encryption never sees a view whose base is the whole parameter (which
        # would keep the plaintext alive for the entire loop).
        enc_chunks.append(crypten.cryptensor(c.contiguous(), **kwargs))
        _trim()
    out = crypten.cat(enc_chunks, dim=0)
    del enc_chunks
    _trim()
    return out


_ORIG_ENCRYPT = _ctm.Module.encrypt


def encrypt(self, mode=True, src=0):
    """Module.encrypt with only the parameter loop replaced (for parameters over the threshold);
    buffers and recursion are the stock body."""
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

print(f"LEVER|encrypt_chunk_cpu|patched|chunks={os.environ.get('SHAFT_ENC_CHUNKS', '1')}|"
      f"threshold={os.environ.get('SHAFT_ENC_CHUNK_THRESHOLD', '5000000')}",
      file=sys.stderr, flush=True)


def _report():
    print(f"LEVER|encrypt_chunk_cpu|chunked_params={_STATS['chunked']}|"
          f"chunks={_STATS['chunks']}|elements={_STATS['elements']}",
          file=sys.stderr, flush=True)


import atexit  # noqa: E402
atexit.register(_report)
