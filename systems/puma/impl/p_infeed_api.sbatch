#!/bin/bash
#SBATCH --job-name=pumaapi
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/pumaapi-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/pumaapi-%j.err

# Prints the infeed API source from the image (introspection only; no measurement, gate or peak).
# Context: x_layer_infeed, 3 replicates, +1.77% regression.
set -uo pipefail

# Apptainer comes from a module on compute nodes; without it the exec silently fails (run 45767572).
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

BASE=/scratch/tmp/j_wred02/snni_campaign/puma
SIF="${SIF:-$BASE/sif/puma-s0_default.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

# PYTHONPATH=/opt/puma makes the module importable; without it the probe fails silently (run 45767678).
apptainer exec --env PYTHONPATH=/opt/puma "$SIF" python3 - <<'PYEOF'
import inspect, re, sys
# THE MODULE IS THE EXAMPLE TREE'S, NOT THE INSTALLED PACKAGE'S. The driver imports
# `examples.python.utils.distributed`, rooted at /opt/puma; `spu.utils.distributed`
# does not exist in this build (runs 45767678 and 45767996 both died on it).
import examples.python.utils.distributed as ppd

src_file = inspect.getsourcefile(ppd)
src = open(src_file).read()
print("SOURCE:", src_file, len(src), "bytes")

print("\n=== module-level names ===")
print(sorted(n for n in dir(ppd) if not n.startswith("_")))

print("\n=== anything that could be a barrier ===")
for n in dir(ppd):
    if re.search(r"wait|barrier|sync|flush|fetch|get|reveal|block", n, re.I):
        o = getattr(ppd, n)
        try:
            print(" ", n, inspect.signature(o))
        except (TypeError, ValueError):
            print(" ", n, type(o).__name__)

print("\n=== what device() returns, and what its call returns ===")
for cls_name in ("Device", "PYU", "SPU", "Object"):
    cls = getattr(ppd, cls_name, None)
    if cls is None:
        continue
    print("\n--- class", cls_name, "---")
    print("  methods:", sorted(m for m in dir(cls) if not m.startswith("__")))
    for meth in ("__call__", "device", "get", "_place", "put"):
        f = getattr(cls, meth, None)
        if f is None or not callable(f):
            continue
        try:
            s = inspect.getsource(f)
        except (OSError, TypeError):
            continue
        # The point is whether the transfer is issued and awaited here or handed to a pool.
        print("  === %s.%s ===" % (cls_name, meth))
        for line in s.splitlines()[:45]:
            print("   ", line)

print("\n=== executor / futures: who actually carries the bytes ===")
for m in re.finditer(r"^.*(ThreadPool|ProcessPool|Future|submit|async|grpc|Serialize|serialize|SerializeToString|tobytes).*$",
                     src, re.M):
    print("  L%-5d %s" % (src[:m.start()].count("\n") + 1, m.group(0)[:120]))
PYEOF
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
