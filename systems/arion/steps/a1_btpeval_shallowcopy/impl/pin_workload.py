#!/usr/bin/env python3
"""Makes ARION's layer and thread counts runtime parameters (environment, upstream defaults) in its
source tree; image build, before `go build`. Scope recorded in RUN_META/markers. Thread count is part
of the measured system (242 -> 139 GiB from 64 to 16 threads).
"""
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/arion")


def edit(path, old, new, what, already, count=1):
    """Apply one patch idempotently; `already` is a required marker present ONLY post-patch (deriving
    it from `new` silently skipped edits in the first version)."""
    p = root / path
    src = p.read_text()
    if already in src:
        print(f"  already patched: {what}")
        return
    if src.count(old) < count:
        sys.exit(f"FAILED: anchor for '{what}' appears {src.count(old)} times in {path}, "
                 f"expected at least {count}. Upstream moved; re-derive the patch.")
    p.write_text(src.replace(old, new, count))
    if already not in p.read_text():
        sys.exit(f"FAILED: applied '{what}' but its marker is still absent from {path}.")
    print(f"  patched: {what}  ({path})")


# --------------------------------------------------------------------------------------------
# A single helper, so both knobs and the marker come from one place and cannot drift apart.
helper = '''package bert

import (
	"fmt"
	"os"
	"strconv"
)

// snniIntEnv returns the value of an environment variable as an int, or def when it is unset or
// unparsable. The value actually used is announced on stderr, because a knob that silently fell
// back to its default looks exactly like one that was applied.
func snniIntEnv(name string, def int) int {
	v := os.Getenv(name)
	if v == "" {
		fmt.Fprintf(os.Stderr, "SNNI_PARAM|%s=%d|source=default\\n", name, def)
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		fmt.Fprintf(os.Stderr, "SNNI_PARAM|%s=%d|source=default (unparsable %q)\\n", name, def, v)
		return def
	}
	fmt.Fprintf(os.Stderr, "SNNI_PARAM|%s=%d|source=env\\n", name, n)
	return n
}
'''
(root / "pkg" / "bert" / "snni_params.go").write_text(helper)
print("  wrote: pkg/bert/snni_params.go")

# The BERT-base multithreaded path is menu option 12 and the only one this campaign measures.
edit(
    "pkg/bert/bertMT.go",
    """	numThreads := 64
	runtime.GOMAXPROCS(numThreads)""",
    """	numThreads := snniIntEnv("ARION_THREADS", 64)
	runtime.GOMAXPROCS(numThreads)""",
    "thread count from ARION_THREADS (default: upstream's 64)",
    already='snniIntEnv("ARION_THREADS"',
    count=2,   # BertMT and BertTinyMT both carry it; both are patched, only the first is measured
)

edit(
    "pkg/bert/bertMT.go",
    """	for i := 0; i < modelParams.NumLayers; i++ {""",
    """	snniLayers := snniIntEnv("ARION_N_LAYERS", modelParams.NumLayers)
	if snniLayers > modelParams.NumLayers {
		snniLayers = modelParams.NumLayers
	}
	fmt.Fprintf(os.Stderr, "WORKLOAD|model=%s|layers=%d|of=%d|threads=%d\\n",
		modelParams.ModelPath, snniLayers, modelParams.NumLayers, numThreads)
	for i := 0; i < snniLayers; i++ {""",
    "layer count from ARION_N_LAYERS (default: the model's own depth)",
    already='snniIntEnv("ARION_N_LAYERS"',
    count=2,
)

# `os` is used by the marker above; bertMT.go does not import it upstream.
edit(
    "pkg/bert/bertMT.go",
    'import (',
    'import (\n\t"os"',
    "bertMT.go imports os",
    already='import (\n\t"os"',
)

print("\n--- randomness in the measured tree (for SEEDS.md, not patched here) ---")
hits = subprocess.run(
    ["grep", "-rn", "-E", r"NewKeyGenerator|NewPRNG|NewKeyedPRNG|crypto/rand|time\.Now\(\)\.Unix",
     "--include=*.go", str(root / "pkg")],
    capture_output=True, text=True).stdout.strip()
print(hits if hits else "(none)")
print("--- end ---\n")
print("PATCH_OK")
