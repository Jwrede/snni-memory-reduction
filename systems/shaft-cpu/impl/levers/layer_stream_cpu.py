"""Model converted to CrypTen one sub-module at a time (ONNX export never holds all of it).
Basis: peak inside torch.onnx.export (VmHWM reaches 99.5% there; four post-export cleanups null). At
the peak the weights exist four times: heap:libstdc++...+0xf4979 867.0 MB (ONNX proto + serialised
string), python3.10+0x13cc27 433.5 MB (returned bytes), libc10.so+0x4a675 433.0 MB (parameters, the
export input); each scales with the exported unit (cf. LLAMA dealer, 9.3%). from_pytorch replaced: split
at mask, embeddings, each layer, head; each piece converted and encrypted alone, run in sequence;
plaintext released by the wrapped s1 lever.
Draws unchanged: (1) all conversion/encryption inside from_pytorch before inference, order = monolithic
order; (2) pieces in execution order = exporter initializer order; (3) seam draw-free (extended mask
(1.0-mask)*finfo.min, both operands public, in its own parameterless graph); (4) wrappers call the real
modules. A moved gate means the split changed draws.
"""

import sys

import torch
import torch.nn as nn

import crypten as ct
from crypten.nn import onnx_converter

# The previously installed from_pytorch, not the pristine one: import AFTER the lever that releases a
# converted model's plaintext, so each PIECE is released as it is converted.
_INNER = onnx_converter.from_pytorch


class _MaskWrap(nn.Module):
    """The encoder's extended attention mask (1.0 - mask[:, None, None, :]) * finfo.min. Inlined
    rather than via BertModel.get_extended_attention_mask, whose wrapper would put every parameter
    back into the export."""

    def __init__(self, dtype):
        super().__init__()
        self._dt = dtype

    def forward(self, attention_mask):
        ext = attention_mask[:, None, None, :].to(dtype=self._dt)
        return (1.0 - ext) * torch.finfo(self._dt).min


class _EmbedWrap(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, input_ids, token_type_ids):
        return self.embeddings(input_ids, token_type_ids=token_type_ids)


class _LayerWrap(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, attention_mask):
        return self.layer(hidden_states, attention_mask=attention_mask)[0]


class _HeadWrap(nn.Module):
    """Pooler, dropout, classifier: the tail of BertForSequenceClassification.forward. Dropout is
    included though an identity in eval mode, because the monolithic trace contains its node."""

    def __init__(self, pooler, dropout, classifier):
        super().__init__()
        self.pooler = pooler
        self.dropout = dropout
        self.classifier = classifier

    def forward(self, hidden_states):
        return self.classifier(self.dropout(self.pooler(hidden_states)))


class StreamedModel:
    """What `from_pytorch` returns: the same interface the measured script uses, over N graphs."""

    def __init__(self, mask_g, embed_g, layer_gs, head_g):
        self._mask = mask_g
        self._embed = embed_g
        self._layers = layer_gs
        self._head = head_g
        self.encrypted = True

    # The script calls encrypt() and then the model. Both must work unchanged.
    def encrypt(self, mode=True, src=0):
        return self

    def decrypt(self):
        raise NotImplementedError("StreamedModel is encrypted piecewise")

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, token_type_ids):
        ext = self._mask(attention_mask)
        hidden = self._embed(input_ids, token_type_ids)
        for g in self._layers:
            hidden = g(hidden, ext)
        return self._head(hidden)


def _convert(mod, dummy, tag):
    """Convert one piece and encrypt it, then report what it cost to hold."""
    g = _INNER(mod, dummy)
    g.encrypt()
    return g


def from_pytorch(pytorch_model, dummy_input, **kwargs):
    bert = getattr(pytorch_model, "bert", None)
    if bert is None or not hasattr(bert, "encoder"):
        print("LEVER|layer_stream_cpu|not a BERT-shaped model, falling back to whole-model conversion",
              file=sys.stderr, flush=True)
        return _INNER(pytorch_model, dummy_input, **kwargs)

    dummy_ids, dummy_mask, dummy_tt = dummy_input
    dtype = next(pytorch_model.parameters()).dtype
    seq = dummy_ids.shape[1]

    # 1. the seam: a parameterless graph for the extended attention mask
    mask_g = _convert(_MaskWrap(dtype), (dummy_mask,), "mask")

    # 2. the embeddings
    embed_g = _convert(_EmbedWrap(bert.embeddings), (dummy_ids, dummy_tt), "embed")
    dummy_hidden = torch.zeros(dummy_ids.shape[0], seq, bert.config.hidden_size, dtype=dtype)
    dummy_ext = torch.zeros(dummy_ids.shape[0], 1, 1, seq, dtype=dtype)

    # 3. each encoder layer, in execution order
    layer_gs = []
    layers = list(bert.encoder.layer)
    for i, layer in enumerate(layers):
        layer_gs.append(_convert(_LayerWrap(layer), (dummy_hidden, dummy_ext), f"layer{i}"))

    # 4. pooler, dropout, classifier
    head_g = _convert(
        _HeadWrap(bert.pooler, pytorch_model.dropout, pytorch_model.classifier),
        (dummy_hidden,), "head",
    )

    print(f"LEVER|layer_stream_cpu|pieces={3 + len(layer_gs)}|layers={len(layer_gs)}",
          file=sys.stderr, flush=True)
    return StreamedModel(mask_g, embed_g, layer_gs, head_g)


onnx_converter.from_pytorch = from_pytorch
ct.nn.from_pytorch = from_pytorch

print("LEVER|layer_stream_cpu|patched", file=sys.stderr, flush=True)
