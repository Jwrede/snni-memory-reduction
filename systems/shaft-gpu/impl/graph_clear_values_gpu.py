"""Drop each graph value as soon as every node that consumes it has run. Source patch of
crypten.nn.module.Graph.forward, ported verbatim from SHAFT-CPU's graph_clear_values_cpu.py. Targets
the parameters CrypTen's executor accumulates AFTER their use (54.7% of the device peak here).
Replaces upstream's disabled cubic _clear_unused_values() with a cheap per-value consumer counter."""
import inspect
import sys
import textwrap

from crypten.nn import module as cnnmod

# graphop_markers_gpu wraps Graph.forward before any lever loads, so inspect.getsource would return
# the wrapper (no anchors). If the marker module is loaded, rewrite the function IT wraps and hand it
# back; otherwise patch the class attribute directly, as SHAFT-CPU does.
_gm = sys.modules.get("graphop_markers_gpu")
if _gm is not None and hasattr(_gm, "_orig_graph_forward"):
    _target = _gm._orig_graph_forward
    def _install(fn):
        _gm._orig_graph_forward = fn
    _where = "graphop_markers_gpu._orig_graph_forward"
else:
    _target = cnnmod.Graph.forward
    def _install(fn):
        cnnmod.Graph.forward = fn
    _where = "crypten.nn.module.Graph.forward"

_SRC = inspect.getsource(_target)

_EDITS = [
    # 1. Build a consumer count per value, once, and remember which values must never be dropped.
    (
        "        computed = {key: False for key in self._graph.keys()}",
        "        computed = {key: False for key in self._graph.keys()}\n"
        "        _pending = {}\n"
        "        for _k, _vl in self._graph.items():\n"
        "            for _n in _vl:\n"
        "                _pending[_n] = _pending.get(_n, 0) + 1\n"
        "        _keep = set(self.output_names)\n"
        "        _ran = None",
    ),
    # 2. Remember which node was just computed, so its inputs can be accounted for below.
    (
        "            module = self._modules[node_to_compute]",
        "            module = self._modules[node_to_compute]\n"
        "            _ran = node_to_compute",
    ),
    # 3. Replace the disabled cubic sweep with the counter, at the same point in the loop.
    (
        "            #_clear_unused_values()",
        "            if _ran is not None:\n"
        "                for _n in self._graph.get(_ran, ()):\n"
        "                    if _n in _pending:\n"
        "                        _pending[_n] -= 1\n"
        "                        if _pending[_n] <= 0 and _n in values and _n not in _keep:\n"
        "                            del values[_n]",
    ),
]

_new = _SRC
for _old, _rep in _EDITS:
    if _old not in _new:
        raise RuntimeError(
            f"graph_clear_values_gpu: expected source not found in Graph.forward: {_old!r}. "
            f"Refusing to patch, because a patch that matched nothing would run as the baseline."
        )
    _new = _new.replace(_old, _rep, 1)

_ns = {}
exec(compile(textwrap.dedent(_new), "<graph_clear_values_gpu>", "exec"), cnnmod.__dict__, _ns)
if "forward" not in _ns:
    raise RuntimeError("graph_clear_values_gpu: rewritten source did not define forward")
_install(_ns["forward"])

# Announcement in this system's form: the sbatch's generic slot greps for `<name>: patched`.
print(f"graph_clear_values_gpu: patched edits={len(_EDITS)} at {_where}", file=sys.stderr, flush=True)
