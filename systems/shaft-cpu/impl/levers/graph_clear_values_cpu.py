"""Each graph value dropped once all its consumers have run.
Object: numpy allocator, 865.8 MB / 39.9% of s6's peak (encrypted parameter shares in numpy buffers;
param_mmap_release_cpu reads them back with np.fromfile); Graph.forward keeps every value for the whole
pass. Upstream's _clear_unused_values (nn/module.py:750-768) is disabled at :918 (cubic scan, 1492
nodes). Here: consumer counts computed once, decremented per node, value deleted at zero unless a graph
output. Source patch of the image's Graph.forward (inspect.getsource, three asserted literal
replacements, exec in the module namespace; a changed image fails). Gate unchanged.
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
]

_new = _SRC
for _old, _rep in _EDITS:
    if _old not in _new:
        raise RuntimeError(
            f"graph_clear_values_cpu: expected source not found in Graph.forward: {_old!r}. "
            f"Refusing to patch, because a patch that matched nothing would run as the baseline."
        )
    _new = _new.replace(_old, _rep, 1)

_ns = {}
exec(compile(textwrap.dedent(_new), "<graph_clear_values_cpu>", "exec"), cnnmod.__dict__, _ns)
if "forward" not in _ns:
    raise RuntimeError("graph_clear_values_cpu: rewritten source did not define forward")
cnnmod.Graph.forward = _ns["forward"]

print(f"LEVER|graph_clear_values_cpu|patched edits={len(_EDITS)}", file=sys.stderr, flush=True)
