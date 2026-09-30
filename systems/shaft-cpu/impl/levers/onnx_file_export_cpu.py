"""ONNX graph exported to a file instead of io.BytesIO.
BytesIO (heap:python3.10+0x13cc27) = 867 MB / 31.7% of s4's peak (2x a 433 MB payload, doubling).
A file serves torch.onnx.export/_load_onnx_model unchanged. The ~433 MB bytes from onnx.load at parse
remain. Gate unchanged. Node-local rootfs (tmpfs refused), not mmap'd.
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
        print(f"FATAL|onnx_file_export_cpu|{d} is on {fs}; that is MEMORY, not disk, so exporting "
              f"there would report a reduction that did not happen", file=sys.stderr, flush=True)
        raise RuntimeError(f"refusing to export onto {fs}")

    # First export is discarded upstream too; a delete-on-close file keeps its buffer out of anon memory.
    with tempfile.NamedTemporaryFile(dir=d, suffix=".onnx", delete=True) as f0:
        onnx_converter._export_pytorch_model(f0, pytorch_model, dummy_input, **kwargs)

    onnx_converter._update_onnx_symbolic_registry()

    f = tempfile.NamedTemporaryFile(dir=d, suffix=".onnx", delete=False)
    atexit.register(lambda p=f.name: os.path.exists(p) and os.unlink(p))
    onnx_converter._export_pytorch_model(f, pytorch_model, dummy_input, **kwargs)
    f.flush()
    size = os.fstat(f.fileno()).st_size
    f.seek(0)
    print(f"LEVER|onnx_file_export_cpu|onnx_bytes={size}|dir={d}|fstype={fs}",
          file=sys.stderr, flush=True)
    # from_pytorch closes this handle after from_onnx parses it, as it did the BytesIO; removed at exit.
    return f


onnx_converter._from_pytorch_to_bytes = _from_pytorch_to_bytes

print("LEVER|onnx_file_export_cpu|patched", file=sys.stderr, flush=True)
