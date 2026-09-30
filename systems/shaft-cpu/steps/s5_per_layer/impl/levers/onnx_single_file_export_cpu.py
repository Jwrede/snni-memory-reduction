"""Export the ONNX graph ONCE, to a file. Removes the buffer doubling and a whole dead export.

SUPERSEDES `onnx_file_export_cpu.py`, which is kept byte-identical so the `lever_md5` in the off-path
run's RUN_META still identifies what ran there. Use one or the other, never both: they patch the same
function.

THE OBJECT is unchanged: `heap:python3.10+0x13cc27`, 867.0 MB and 31.6% of s4's peak, CPython's allocator
serving `io.BytesIO`. It is 867 MB for a 433 MB payload because a BytesIO grows by doubling.

WHY A SECOND ATTEMPT ON IT. `off-path/s5_onnx_file_export/` moved the peak -110820 kB (-4.1%) by sending
the bytes to a file instead, and cost +3.4% wall against a 0.94% wall spread: `cost_type = paid`. The
memory was right and the runtime was avoidable, because that version kept doing something the stock code
does that is pure waste here.

THE DEAD EXPORT. `_from_pytorch_to_bytes` (`nn/onnx_converter.py:121-130`) exports the whole model to
ONNX, throws the result away, calls `_update_onnx_symbolic_registry()`, and exports again. The comment
says the first export exists "only ... to obtain the PyTorch-to-ONNX symbolic registry". That is true of
the OLD torch path, where the update walks `torch.onnx.symbolic_registry._registry` and therefore needs
it to have been populated. This image is torch 2.0.1, where that module does not exist:

    >>> import torch.onnx.symbolic_registry
    ImportError: No module named 'torch.onnx.symbolic_registry'

so `SYM_REGISTRY` is False (`nn/onnx_converter.py:31-38`) and `_update_onnx_symbolic_registry` takes its
`else` branch, which is six `torch.onnx.register_custom_op_symbolic` calls (`:307-326`). Those register
by name and need no prior export at all. **On this torch version the first export is dead work**: a full
model trace and serialisation whose only output is discarded.

So this lever exports once, to a file. The saving is the buffer doubling AND one entire export's runtime,
which is why it should be free where the file-only variant was paid.

THE GUARD MATTERS. If `SYM_REGISTRY` is True -- an older torch -- the first export IS load-bearing and
this lever keeps it, falling back to two exports with the file. The claim above is about this image, and
the code checks rather than assumes it.

WHY THE GATE CANNOT MOVE: the second export is the one whose bytes were ever used, it runs with the same
registry updates applied before it, with the same kwargs, on the same model, and `onnx.load` parses the
same bytes. No generator is consulted; the model is in eval mode, so its Dropout modules draw nothing.

NOT A FAKE REDUCTION: node-local rootfs, verified from /proc/mounts and logged, tmpfs refused; and the
file is not mmap'd, because a mapped file would be counted as `file:` and excluded from `W` as library
overhead, moving bytes out of the number rather than out of memory.
"""

import atexit
import io
import os
import sys
import tempfile

from crypten.nn import onnx_converter

_ORIG = onnx_converter._from_pytorch_to_bytes
_BAD_FS = ("tmpfs", "ramfs", "devtmpfs")


def _fstype(path):
    best, best_len = "unknown", -1
    real = os.path.realpath(path)
    try:
        for ln in open("/proc/mounts"):
            f = ln.split()
            if len(f) < 3:
                continue
            mp = f[1]
            if (real == mp or real.startswith(mp.rstrip("/") + "/")) and len(mp) > best_len:
                best, best_len = f[2], len(mp)
    except OSError:
        pass
    return best


def _from_pytorch_to_bytes(pytorch_model, dummy_input, **kwargs):
    d = os.environ.get("SNNI_SPILL_DIR", "/tmp")
    fs = _fstype(d)
    if fs in _BAD_FS:
        print(f"FATAL|onnx_single_file_export_cpu|{d} is on {fs}; that is MEMORY, not disk, so "
              f"exporting there would report a reduction that did not happen",
              file=sys.stderr, flush=True)
        raise RuntimeError(f"refusing to export onto {fs}")

    two_exports = bool(getattr(onnx_converter, "SYM_REGISTRY", False))
    if two_exports:
        # Older torch: the registry update walks a registry that an export populates, so the first
        # export is load-bearing and is kept. It still goes to a file rather than a BytesIO.
        with tempfile.NamedTemporaryFile(dir=d, suffix=".onnx", delete=True) as f0:
            onnx_converter._export_pytorch_model(f0, pytorch_model, dummy_input, **kwargs)

    onnx_converter._update_onnx_symbolic_registry()

    f = tempfile.NamedTemporaryFile(dir=d, suffix=".onnx", delete=False)
    atexit.register(lambda p=f.name: os.path.exists(p) and os.unlink(p))
    onnx_converter._export_pytorch_model(f, pytorch_model, dummy_input, **kwargs)
    f.flush()
    size = os.fstat(f.fileno()).st_size
    f.seek(0)
    print(f"LEVER|onnx_single_file_export_cpu|onnx_bytes={size}|exports="
          f"{2 if two_exports else 1}|sym_registry={two_exports}|dir={d}|fstype={fs}",
          file=sys.stderr, flush=True)
    # `from_pytorch` closes this handle after `from_onnx` has parsed it, exactly as it closed the
    # BytesIO, so the calling contract is unchanged. The file is removed at exit.
    return f


onnx_converter._from_pytorch_to_bytes = _from_pytorch_to_bytes

print("LEVER|onnx_single_file_export_cpu|patched", file=sys.stderr, flush=True)
