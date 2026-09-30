"""Exports the ONNX graph once, to a file, instead of twice into io.BytesIO (port of SHAFT-CPU's
onnx_single_file_export_cpu.py, s4_onnx_and_spill). Target: 870.5 MB heap row at s2_embed_chunk's
peak (BytesIO doubling, 2x the 433 MB payload). Refuses tmpfs/ramfs/devtmpfs (/proc/mounts)."""
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
        print(f"FATAL|onnx_single_file_export_gpu|{d} is on {fs}; that is MEMORY, not disk, so "
              f"exporting there would report a reduction that did not happen",
              file=sys.stderr, flush=True)
        raise RuntimeError(f"refusing to export onto {fs}")

    two_exports = bool(getattr(onnx_converter, "SYM_REGISTRY", False))
    if two_exports:
        # Older torch: the first export populates the symbolic registry, so it is load-bearing and
        # kept (to a file, not a BytesIO).
        with tempfile.NamedTemporaryFile(dir=d, suffix=".onnx", delete=True) as f0:
            onnx_converter._export_pytorch_model(f0, pytorch_model, dummy_input, **kwargs)

    onnx_converter._update_onnx_symbolic_registry()

    f = tempfile.NamedTemporaryFile(dir=d, suffix=".onnx", delete=False)
    atexit.register(lambda p=f.name: os.path.exists(p) and os.unlink(p))
    onnx_converter._export_pytorch_model(f, pytorch_model, dummy_input, **kwargs)
    f.flush()
    size = os.fstat(f.fileno()).st_size
    f.seek(0)
    print(f"LEVER|onnx_single_file_export_gpu|onnx_bytes={size}|exports="
          f"{2 if two_exports else 1}|sym_registry={two_exports}|dir={d}|fstype={fs}",
          file=sys.stderr, flush=True)
    # from_pytorch closes this handle after from_onnx parses it, as it closed the BytesIO; removed at exit.
    return f


onnx_converter._from_pytorch_to_bytes = _from_pytorch_to_bytes

# Announcement in this system's form (not the donor's LEVER|<name>|patched): the generic slot greps
# stderr for `<name>: patched` and exits 5 when it is missing.
print("onnx_single_file_export_gpu: patched", file=sys.stderr, flush=True)
