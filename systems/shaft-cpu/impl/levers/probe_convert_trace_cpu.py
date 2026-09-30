"""Diagnostic only. Brackets ONNX-to-CrypTen conversion (after s7 all replicates peak between
after_model_load and after_model_convert; numpy allocator 865.8 MB, ~2x 433 MB weights). _to_crypten
builds one tensor per initializer (torch.from_numpy) while the ModelProto holds raw_data. Reports RSS
and live torch tensor bytes per stage. Never in a measured run.
"""

import gc
import os
import sys

import torch

from crypten.nn import onnx_converter

_OUT = None
_ORIG_TO_CRYPTEN = onnx_converter._to_crypten
_ORIG_LOAD = onnx_converter._load_onnx_model


def _rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def _torch_bytes():
    """Total bytes of live torch tensor storages, deduplicated by data pointer."""
    seen = {}
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor):
                continue
            st = obj.untyped_storage()
            seen[st.data_ptr()] = st.nbytes()
        except Exception:
            continue
    return len(seen), sum(seen.values())


def _mark(tag):
    n, b = _torch_bytes()
    print(f"CONVERT|{tag}|rss_kb={_rss_kb()}|torch_tensors={n}|torch_bytes={b}",
          file=_OUT, flush=True)


def _load_onnx_model(x):
    _mark("before_parse")
    m = _ORIG_LOAD(x)
    _mark("after_parse")
    return m


def _to_crypten(onnx_model):
    from onnx import numpy_helper
    from crypten.nn import module

    _mark("to_crypten_entry")
    n_init = len(onnx_model.graph.initializer)
    raw = sum(len(i.raw_data) for i in onnx_model.graph.initializer)
    print(f"CONVERT|initializers|count={n_init}|raw_data_bytes={raw}", file=_OUT, flush=True)

    input_names, output_names = onnx_converter._get_input_output_names(onnx_model)
    crypten_model = module.Graph(input_names, output_names[0])

    for i, node in enumerate(onnx_model.graph.initializer):
        param = torch.from_numpy(numpy_helper.to_array(node))
        crypten_model.add_module(node.name, module.Parameter(param), [])
        if i in (0, n_init // 4, n_init // 2, (3 * n_init) // 4, n_init - 1):
            _mark(f"initializer_{i}_of_{n_init}")

    for node in onnx_model.graph.node:
        attributes = {a.name: onnx_converter._get_attribute_value(a) for a in node.attribute}
        cls = onnx_converter._get_operator_class(node.op_type, attributes)
        m = cls.from_onnx(attributes=attributes)
        ins = list(node.input)
        outs = list(node.output)
        if node.op_type == "Dropout":
            outs = [outs[0]]
        crypten_model.add_module(outs[0], m, ins, output_names=outs)
    _mark("after_nodes")

    crypten_model = onnx_converter._get_model_or_module(crypten_model)
    _mark("to_crypten_exit")
    return crypten_model


def _start():
    global _OUT
    pid = os.getpid()
    d = os.environ.get("RESULTS_DIR", "/tmp")
    _OUT = open(os.path.join(d, f"converttrace_pid{pid}.txt"), "w", buffering=1)
    print(f"LEVER|probe_convert_trace_cpu|patched pid={pid}", file=sys.stderr, flush=True)


_start()
onnx_converter._to_crypten = _to_crypten
onnx_converter._load_onnx_model = _load_onnx_model
