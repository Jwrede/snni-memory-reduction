"""Parameter materialised only when nothing else can run (borrowed from SHAFT-GPU s4_param_defer:
VRAM 1,071,104 -> 362,496 kB, -66.2%, byte-identical). Parameter nodes have no inputs: upstream
evaluates all before any operator, defeating spill and graph_clear. Transfer test (published peak is
the ONNX conversion; the per-layer arm loads one layer at a time). = graph_clear_values_cpu.py plus one
edit (carries its three edits; both rewrite Graph.forward).
"""
import inspect
import sys
import textwrap

from crypten.nn import module as cnnmod

_SRC = inspect.getsource(cnnmod.Graph.forward)

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
    # 4. PARAMETERS LAST, borrowed verbatim from SHAFT-GPU's `param_defer_gpu`.
    (
        """        def _find_computable_node():
            \"\"\"Find a node for which all inputs are available.\"\"\"
            for key, inputs_available_list in inputs_available.items():
                if all(inputs_available_list) and not computed[key]:
                    return key
            return None""",
        """        def _find_computable_node():
            \"\"\"Find a node for which all inputs are available, PARAMETERS LAST.\"\"\"
            _deferred = None
            for key, inputs_available_list in inputs_available.items():
                if all(inputs_available_list) and not computed[key]:
                    if type(self._modules.get(key)).__name__ == "Parameter":
                        if _deferred is None:
                            _deferred = key
                        continue
                    return key
            return _deferred""",
    ),
]

_new = _SRC
for _old, _rep in _EDITS:
    if _old not in _new:
        raise RuntimeError(
            f"param_defer_cpu: expected source not found in Graph.forward: {_old!r}. "
            f"Refusing to patch, because a patch that matched nothing would run as the baseline."
        )
    _new = _new.replace(_old, _rep, 1)

_ns = {}
exec(compile(textwrap.dedent(_new), "<param_defer_cpu>", "exec"), cnnmod.__dict__, _ns)
if "forward" not in _ns:
    raise RuntimeError("param_defer_cpu: rewritten source did not define forward")
cnnmod.Graph.forward = _ns["forward"]

print(f"LEVER|param_defer_cpu|patched edits={len(_EDITS)}", file=sys.stderr, flush=True)
