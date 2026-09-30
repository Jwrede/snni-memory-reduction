"""Large parameters encrypted in slices, each slice written straight into the finished share.
Supersedes encrypt_chunk_inplace_cpu (s8_encrypt_chunk unchanged for reproducibility); same lever
without the final concatenation. p_graphop_s8: 91.8% of the peak present before any graph node, inside
enc_embed.encrypt(); s8 reduced the transient 601,612 -> 384,348 kB (-36%). crypten.cat holds eight
173,976 kB inputs and the 173,976 kB output (~347,952 kB; measured 384,348). Output share allocated once
at the first chunk; later chunks copied into their row range and freed: full output 173,976 + one chunk
21,747 = ~195,723 kB. Same eight encryptions, order, generator, bytes, row order. Free. Gate expected
identical to s8's (a misordered slice would permute the table; the gate detects it).
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
    print("FATAL|encrypt_chunk_inplace_cpu|encrypt_spill_cpu is already loaded, so it has captured the "
          "unchunked Module.encrypt and this lever would never run. Put encrypt_chunk_inplace_cpu BEFORE "
          "encrypt_spill_cpu in SHAFT_LEVERS.", file=sys.stderr, flush=True)
    raise RuntimeError("encrypt_chunk_inplace_cpu must load before encrypt_spill_cpu")


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
    total_rows = plaintext.shape[0]
    out = None
    filled = 0
    for c in splits:
        # .contiguous() so encryption never sees a view whose base is the whole parameter (which
        # would keep the plaintext alive for the entire loop).
        enc = crypten.cryptensor(c.contiguous(), **kwargs)
        inner = enc._tensor if hasattr(enc, "_tensor") else enc
        rows = inner.share.shape[0]

        if out is None:
            # Output allocated once here; the first chunk's own wrapper is kept as the result, so its
            # class, encoder and scale come from a real encryption rather than a reconstruction.
            full = torch.empty((total_rows,) + tuple(inner.share.shape[1:]),
                               dtype=inner.share.dtype)
            full[0:rows] = inner.share
            inner.share = full
            out = enc
            filled = rows
        else:
            outer = out._tensor if hasattr(out, "_tensor") else out
            outer.share[filled:filled + rows] = inner.share
            filled += rows
            del enc, inner
        _trim()

    if filled != total_rows:
        # torch.chunk may return fewer chunks than asked; a short fill leaves uninitialised rows
        # (plausible logits, wrong answer).
        raise RuntimeError(f"encrypt_chunk_inplace_cpu: filled {filled} of {total_rows} rows")
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

print(f"LEVER|encrypt_chunk_inplace_cpu|patched|chunks={os.environ.get('SHAFT_ENC_CHUNKS', '1')}|"
      f"threshold={os.environ.get('SHAFT_ENC_CHUNK_THRESHOLD', '5000000')}",
      file=sys.stderr, flush=True)


def _report():
    print(f"LEVER|encrypt_chunk_inplace_cpu|chunked_params={_STATS['chunked']}|"
          f"chunks={_STATS['chunks']}|elements={_STATS['elements']}",
          file=sys.stderr, flush=True)


import atexit  # noqa: E402
atexit.register(_report)
