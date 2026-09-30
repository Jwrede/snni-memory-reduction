"""Diagnostic only. RSS and VmHWM after every statement of the encrypted embedding (peak reached
before the Beaver matmul: one-hot pair, masked-index reveal, two (1,128,V) index tensors, gather).
Reimplements Embedding.forward statement for statement with marks. Never in a measured run.
"""

import os
import sys

import torch

import crypten
from crypten.nn import module as cnnmod

_OUT = None
_ORIG_FORWARD = cnnmod.Embedding.forward


def _stat(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1])
    return -1


def _mark(tag):
    print(f"EMBFWD|{tag}|rss_kb={_stat('VmRSS:')}|hwm_kb={_stat('VmHWM:')}",
          file=_OUT, flush=True)


def forward(self, input):
    weight, indices, _ = input
    if not crypten.is_encrypted_tensor(indices):
        return _ORIG_FORWARD(self, input)

    voc_size = weight.shape[0]
    _mark("entry")

    provider = crypten.mpc.get_default_provider()
    r, v = provider.generate_one_hot_pair(indices.size(), voc_size, device=indices.device)
    _mark("after_one_hot_pair")

    j = indices - r
    _mark("after_mask_index")
    j = j.get_plain_text().int().unsqueeze(-1) % voc_size
    _mark("after_reveal_index")

    arange1 = torch.arange(voc_size, device=indices.device).view([1, 1, voc_size]).repeat(
        (indices.shape[0], indices.shape[1], 1)
    )
    _mark("after_arange1")
    arange2 = (arange1 - j) % voc_size
    _mark("after_arange2")

    idx = v.gather(2, arange2)
    _mark("after_gather")

    output = idx.matmul(weight)
    _mark("after_matmul")
    return output


def _start():
    global _OUT
    pid = os.getpid()
    d = os.environ.get("RESULTS_DIR", "/tmp")
    _OUT = open(os.path.join(d, f"embfwd_pid{pid}.txt"), "w", buffering=1)
    print(f"LEVER|probe_embed_forward_cpu|patched pid={pid}", file=sys.stderr, flush=True)


_start()
cnnmod.Embedding.forward = forward
