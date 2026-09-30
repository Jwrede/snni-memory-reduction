"""ONNX graph exported once, to a file (no buffer doubling, no dead second export).
Supersedes onnx_file_export_cpu.py (byte-identical for its lever_md5); use one. File-only variant
(off-path/s5_onnx_file_export) paid (+3.4% wall): _from_pytorch_to_bytes still exported twice. The first
export is needed only on the old torch.onnx.symbolic_registry path, absent in torch 2.0.1
(SYM_REGISTRY False); a SYM_REGISTRY guard keeps two exports on older torch. Gate unchanged. Node-local
rootfs, tmpfs refused, not mmap'd.
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
        # Older torch: the first export populates the registry the update walks, so it is kept (to a file).
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
    # from_pytorch closes this handle after from_onnx parses it, as it did the BytesIO; removed at exit.
    return f


onnx_converter._from_pytorch_to_bytes = _from_pytorch_to_bytes

print("LEVER|onnx_single_file_export_cpu|patched", file=sys.stderr, flush=True)
