"""Keeps encrypted parameters on the host; each is on the device only during its use.
Target: 866.6 MB whole encrypted model in VRAM at s1's peak (35.2% of 2.29 GiB; CrypTen holds every
value until the pass ends). Patches: Parameter ignores .cuda(); other modules' forward get a device
copy of host-resident encrypted inputs, dropped on return. Risk: host rise equal to the VRAM drop =
placement (cross_pool=yes)."""
import os
import sys

import crypten
import crypten.nn.module as _ctm

_ENABLED = os.environ.get("SHAFT_PARAM_STREAM", "0") == "1"
_MARK = "param_stream_gpu: patched"


def _host_encrypted(x):
    """True for an encrypted tensor not on the device. After the Parameter patch below that is
    exactly the parameters, since every activation is produced on the device by the prior operator."""
    if not crypten.is_encrypted_tensor(x):
        return False
    try:
        return not x.is_cuda
    except Exception:
        return False


def _map(obj, fn):
    """Rebuild a value, list or tuple of values with fn applied to each leaf."""
    if isinstance(obj, list):
        return [_map(item, fn) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_map(item, fn) for item in obj)
    return fn(obj)


def _wrap_forward(mod):
    # crypten's Module.__getattribute__ re-resolves 'forward' at call time, so capturing `mod.forward`
    # would recurse into this wrapper. Take the real bound method (as graphop_markers_gpu does).
    orig = object.__getattribute__(mod, "forward")

    def forward(*args, **kwargs):
        # A device copy is handed to the operator; the original is never touched. Moving the
        # parameter itself and back (cheaper by one copy) produced WRONG OUTPUT (gate 3.157e+09 vs
        # baseline 1.962e-01): an operator may return a value sharing its input's storage, and
        # .cuda()/.cpu() round trips can leave a CUDALongTensor wrapper holding host data.
        def _dev(x):
            if not _host_encrypted(x):
                return x
            y = x.clone()      # deep copy on the host: `shallow_copy` shares `_tensor`, and
            y.cuda()           # `MPCTensor.share` writes through, so it would move the original
            return y

        args = tuple(_map(a, _dev) for a in args)
        kwargs = {k: _map(v, _dev) for k, v in kwargs.items()}
        return orig(*args, **kwargs)
        # The device copies die with this frame unless the operator kept a reference, in which case
        # they belong to the output rather than the parameter.

    mod.forward = forward
    mod._param_stream_wrapped = True


def _patch():
    # (1) a Parameter module ignores .cuda(): its share stays on the host.
    def _parameter_cuda(self, device=None):
        return self

    _ctm.Parameter.cuda = _parameter_cuda

    # (2) every other module gets its host-resident encrypted inputs on the device around its
    # forward. Wrapped per instance at the first Graph.forward, once.
    _orig_graph_forward = _ctm.Graph.forward

    def _streaming_forward(self, *args, **kwargs):
        if not getattr(self, "_param_stream_instrumented", False):
            for name, mod in self._modules.items():
                if isinstance(mod, (_ctm.Parameter, _ctm.Constant)):
                    continue
                if getattr(mod, "_param_stream_wrapped", False):
                    continue
                _wrap_forward(mod)
            self._param_stream_instrumented = True
        return _orig_graph_forward(self, *args, **kwargs)

    _ctm.Graph.forward = _streaming_forward


if _ENABLED:
    _patch()
    # The harness fails the run if this announcement is absent (a silent lever publishes the baseline).
    print(_MARK, file=sys.stderr, flush=True)
else:
    print("param_stream_gpu: inactive (SHAFT_PARAM_STREAM != 1)", file=sys.stderr, flush=True)
