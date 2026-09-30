"""Defers each Parameter's materialisation until no other node is computable. Parameter nodes have
no inputs, so upstream's scheduler evaluates all 201 before any operator (each a disk reload via
param_forward); graph_clear_values_gpu cannot help then. Free (same nodes and order constraints; a
Parameter evaluation is a disk read); gate identical to s3's. = graph_clear_values_gpu.py plus one
edit (carries its three edits; both rewrite the same function)."""
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
    # 4. PARAMETERS LAST. A `Parameter` node has no inputs, so it is computable from the first
    #    iteration, and the initializers come first in the graph: without this every parameter is
    #    materialised before any operator runs.
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
            f"param_defer_gpu: expected source not found in Graph.forward: {_old!r}. "
            f"Refusing to patch, because a patch that matched nothing would run as the baseline."
        )
    _new = _new.replace(_old, _rep, 1)

_ns = {}
exec(compile(textwrap.dedent(_new), "<param_defer_gpu>", "exec"), cnnmod.__dict__, _ns)
if "forward" not in _ns:
    raise RuntimeError("param_defer_gpu: rewritten source did not define forward")
_install(_ns["forward"])

# Announcement in this system's form: the sbatch's generic slot greps for `<name>: patched`.
print(f"param_defer_gpu: patched edits={len(_EDITS)} at {_where}", file=sys.stderr, flush=True)
