"""Encrypt a large parameter in slices, so its three simultaneous copies are one Nth each.

THE OBJECT, and for once it is not read off an object table but off the CLOCK. `s7_embed_chunk`'s
published peak is 1,080,140 kB (VmHWM) while the highest PHASE MARKER in the same run is 393,464.
The peak is 2.7x any marker, so it is a transient between two of them, and the poller says which
two:

    embed|after_convert     393,464 kB      t = -1,416 ms
    POLLER MAXIMUM          995,076 kB      t =      0
    embed|after_encrypt     320,360 kB      t =   +288 ms

**The peak of this whole line is inside one call to `enc_embed.encrypt()`**, and it is a 601,612 kB
transient that no marker sees and no object table names. This is what `anon:residual` -- 711.6 MB,
64.3% of the endpoint -- has been all along.

WHY IT IS THAT BIG, and the donor did the arithmetic before we measured it. Encrypting one parameter
holds, at the same instant:

    the plaintext                      94 MB   float32, 28996 x 768
    its int64 share                   187 MB
    the random mask share             187 MB
    the share to send                 187 MB
    ------------------------------------------
                                     ~570 MB

Measured here: 601,612 kB. The prediction was made in the retired campaign's own comment and is
within 5% of what this campaign's poller found four months later.

THE MECHANISM, taken from that campaign's `08_low_cpu_mem_load/bert_instrumented.py`: split any
parameter above a threshold into N chunks along dim 0, encrypt each, collect, `crypten.cat` them
back, and collect plus `malloc_trim(0)` between chunks. With N=8 the donor's comment puts the
transient at about 70 MB.

**ONLY THE EMBEDDING QUALIFIES, and that is deliberate rather than a limitation.** The threshold is
5,000,000 elements, the embedding is 28,996 x 768 = 22.3M, and the next largest parameter is
768 x 3072 = 2.36M, less than half the threshold. So this lever touches exactly one parameter, and
it is the one the poller caught.

LOAD ORDER IS A HARD REQUIREMENT, not a preference. `encrypt_spill_cpu` wraps `Module.encrypt` and
calls the original first, then spills the finished share. This lever REPLACES `Module.encrypt`, so
it has to be in place before the spill lever captures its original, or the chunking is patched over
and silently never runs -- a lever that never loaded is a successful run with a zero delta and no
error. The check below refuses rather than trusting the `SHAFT_LEVERS` order to be right.

FREE, first category: the same bytes are encrypted from the same plaintext in the same order, one
slice at a time instead of all at once. Nothing is recomputed and no plaintext is re-read.

PHASE B IS EXPECTED, and the reason is the same one `embed_chunk_cpu` has. `crypten.cryptensor(x,
src=0)` broadcasts a share from the source party, so eight slices are eight broadcasts where there
was one: the SAME bytes in more rounds, and the mask stream is drawn in eight pieces instead of one,
which moves the output. The line has been in phase B since `s5`, so this states no new kind of claim.
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

# THE ORDER CHECK. `encrypt_spill_cpu` captures `Module.encrypt` at ITS import time; if it has
# already done so, replacing the method now leaves the spill lever calling the stock one and this
# lever does nothing at all while reporting that it patched.
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
        # `.contiguous()` matters: a chunk of a 2-D tensor along dim 0 is already contiguous, but
        # making it explicit means the encryption never sees a view whose base is the whole
        # parameter, which would keep the plaintext alive for the entire loop.
        enc_chunks.append(crypten.cryptensor(c.contiguous(), **kwargs))
        _trim()
    out = crypten.cat(enc_chunks, dim=0)
    del enc_chunks
    _trim()
    return out


_ORIG_ENCRYPT = _ctm.Module.encrypt


def encrypt(self, mode=True, src=0):
    """`Module.encrypt` with the parameter loop replaced; everything else is the stock body.

    The stock method walks this module's own parameters, then its buffers, then recurses. Only the
    first of those three is changed, and only for parameters over the threshold.
    """
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
