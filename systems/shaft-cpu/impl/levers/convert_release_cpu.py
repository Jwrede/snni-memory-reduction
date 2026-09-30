"""Plaintext weights held once during conversion, and the freed pages returned.
Supersedes encrypt_on_convert_cpu (kept byte-identical for its lever_md5). Target: 865.8 MB of
plaintext weights held twice during ONNX-to-CrypTen conversion (s7's peak); three earlier attempts
removed the object, not the peak (glibc reuse). (1) encrypt each initializer in its creating loop
(upstream Module.encrypt; one plaintext weight live); (2) mallopt(M_MMAP_THRESHOLD, 131072) for
from_onnx, restored to 33554432, then malloc_trim(0). Draws unchanged (same cryptensor calls, tensors
and order; insertion order = initializer order). Both parts report their effect.
"""


import sys

import torch

import crypten
from crypten.nn import module
from crypten.nn import onnx_converter

import ctypes

_ORIG = onnx_converter._to_crypten
_ORIG_FROM_ONNX = onnx_converter.from_onnx
_N = 0

_M_MMAP_THRESHOLD = -3          # glibc malloc.h
_LOW = 131072                   # during from_onnx: mmap the big blocks so free() munmaps them
_RESTORE = 33554432             # afterwards: where glibc's own dynamic threshold tops out
try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError as _exc:         # pragma: no cover
    _LIBC = None
    print(f"LEVER|convert_release_cpu|libc unavailable: {_exc}", file=sys.stderr, flush=True)


def _rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def _mallopt(value, tag):
    if _LIBC is None:
        return
    rc = _LIBC.mallopt(_M_MMAP_THRESHOLD, ctypes.c_int(value))
    print(f"LEVER|convert_release_cpu|mallopt M_MMAP_THRESHOLD={value} rc={rc} ({tag})",
          file=sys.stderr, flush=True)


def from_onnx(onnx_string_or_file):
    """The window in which the plaintext weights are created and released."""
    _mallopt(_LOW, "entering from_onnx")
    try:
        return _ORIG_FROM_ONNX(onnx_string_or_file)
    finally:
        _mallopt(_RESTORE, "leaving from_onnx")
        if _LIBC is not None:
            before = _rss_kb()
            rc = _LIBC.malloc_trim(0)
            after = _rss_kb()
            print(f"LEVER|convert_release_cpu|malloc_trim rc={rc}|rss_kb {before} -> {after}|"
                  f"returned_kb={before - after}", file=sys.stderr, flush=True)


def _to_crypten(onnx_model):
    global _N
    from onnx import numpy_helper

    input_names, output_names = onnx_converter._get_input_output_names(onnx_model)
    assert len(output_names) == 1, "Only one output per model supported."
    crypten_model = module.Graph(input_names, output_names[0])

    _N = 0
    for node in onnx_model.graph.initializer:
        param = torch.from_numpy(numpy_helper.to_array(node))
        p = module.Parameter(param)
        del param
        # Call upstream's own per-module encrypt here (same cryptensor call the deferred encrypt would
        # issue, same arguments), dropping the plaintext as soon as it returns. Must go through
        # Parameter() first: register_parameter reads param.dtype (nn/module.py:129) and an MPCTensor
        # has none, so a Parameter built directly from an encrypted tensor raises; set_parameter (what
        # encrypt uses) has no such check.
        p.encrypt(mode=True, src=0)
        crypten_model.add_module(node.name, p, [])
        _N += 1

    for node in onnx_model.graph.node:
        attributes = {a.name: onnx_converter._get_attribute_value(a) for a in node.attribute}
        crypten_class = onnx_converter._get_operator_class(node.op_type, attributes)
        crypten_module = crypten_class.from_onnx(attributes=attributes)
        ins = list(node.input)
        outs = list(node.output)
        if node.op_type == "Dropout":
            outs = [outs[0]]           # do not output Dropout mask
        crypten_model.add_module(outs[0], crypten_module, ins, output_names=outs)

    crypten_model = onnx_converter._get_model_or_module(crypten_model)
    print(f"LEVER|convert_release_cpu|encrypted_initializers={_N}", file=sys.stderr, flush=True)
    return crypten_model


onnx_converter._to_crypten = _to_crypten
onnx_converter.from_onnx = from_onnx

print("LEVER|convert_release_cpu|patched", file=sys.stderr, flush=True)
