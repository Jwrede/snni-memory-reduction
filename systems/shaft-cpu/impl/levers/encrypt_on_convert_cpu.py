"""Each ONNX initializer encrypted as it is converted (plaintext weight set never accumulates).
Supersedes plaintext_free_early_cpu for its object (may load together). Object: plaintext weights in
_to_crypten, 865.8 MB at s7's peak = 2 x 433.25 MB (HF model + per-initializer copies from
torch.from_numpy, nn/onnx_converter.py:183); s8_plaintext_free_early released the first, peak
unchanged. Remaining 433 MB = all 201 initializers alive at once (graph built plaintext, encrypt()
afterwards); encrypting in the creating loop keeps one live. Draws unchanged (Module.encrypt walks
self._modules in insertion order = initializer order). Later encrypt() skips already-encrypted
parameters (Parameter.__init__ sets self.encrypted = is_encrypted_tensor(param), nn/module.py:1126;
skip at :478) and still sets every flag. Used with plaintext_free_early_cpu; alone it moved the peak 0.06%.
"""

import sys

import torch

import crypten
from crypten.nn import module
from crypten.nn import onnx_converter

_ORIG = onnx_converter._to_crypten
_N = 0


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
    print(f"LEVER|encrypt_on_convert_cpu|encrypted_initializers={_N}", file=sys.stderr, flush=True)
    return crypten_model


onnx_converter._to_crypten = _to_crypten

print("LEVER|encrypt_on_convert_cpu|patched", file=sys.stderr, flush=True)
