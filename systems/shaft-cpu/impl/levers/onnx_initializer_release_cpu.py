"""Each ONNX initializer dropped from the protobuf once converted to a tensor.
At s5_embed_chunk's peak the protobuf parse is rank 1 (867.0 MB, 42.2%; 2x the 433 MB payload: each
weight in the message and as the bytes its tensor views). First category (retained past last use).
.copy() guard required (to_array may return a view the protobuf owns). malloc_trim(0) included.
Asserted source patch of _to_crypten via inspect.getsource (a changed image fails).
"""

import ctypes
import inspect
import sys

from crypten.nn import onnx_converter as _oc

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None

_SRC = inspect.getsource(_oc._to_crypten)

_EDITS = [
    (
        "    for node in onnx_model.graph.initializer:\n"
        "        param = torch.from_numpy(numpy_helper.to_array(node))\n"
        "        crypten_model.add_module(node.name, module.Parameter(param), [])",

        "    _released = 0\n"
        "    for node in onnx_model.graph.initializer:\n"
        "        _arr = numpy_helper.to_array(node)\n"
        "        # Own the buffer before the field it may be viewing is cleared.\n"
        "        if not _arr.flags.owndata:\n"
        "            _arr = _arr.copy()\n"
        "        param = torch.from_numpy(_arr)\n"
        "        crypten_model.add_module(node.name, module.Parameter(param), [])\n"
        "        # LAST USE. The tensor exists and owns its bytes; the protobuf copy has no reader.\n"
        "        if node.HasField('raw_data'):\n"
        "            _released += len(node.raw_data)\n"
        "            node.ClearField('raw_data')\n"
        "    del onnx_model.graph.initializer[:]\n"
        "    _snni_release_report(_released)",
    ),
]

_new = _SRC
for _old, _rep in _EDITS:
    if _old not in _new:
        print("LEVER|onnx_initializer_release_cpu|FAILED: the initializer loop in "
              "onnx_converter._to_crypten does not match this lever's expectation; refusing to "
              "patch nothing and report success", file=sys.stderr, flush=True)
        raise RuntimeError("onnx_initializer_release_cpu: anchor not found")
    _new = _new.replace(_old, _rep, 1)


def _snni_release_report(released):
    """Return the freed pages to the kernel and say how many bytes were dropped."""
    import gc
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass
    print(f"LEVER|onnx_initializer_release_cpu|released_protobuf_bytes={released}",
          file=sys.stderr, flush=True)


_oc.__dict__["_snni_release_report"] = _snni_release_report
exec(compile(_new, "<onnx_initializer_release_cpu>", "exec"), _oc.__dict__)

print("LEVER|onnx_initializer_release_cpu|patched", file=sys.stderr, flush=True)
