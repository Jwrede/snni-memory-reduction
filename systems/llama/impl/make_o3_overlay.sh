#!/bin/bash
# Build the o3 overlay: o2's files plus an api.cpp whose two leaking functions (SlothGelu, SlothARS)
# release what they allocate.  make_o3_overlay.sh (run on the cluster, in the llama campaign dir)
# Functions are located by extent (signature in column 0, next column-0 brace closes it), not by
# anchor text, since both end with the same four lines. Refuses rather than guesses.
set -uo pipefail

B=/scratch/tmp/j_wred02/snni_campaign/llama
SIF="$B/sif/llama-online-o2_activation_free.sif"
SRC=/opt/EzPC/GPU-MPC/ext/sytorch/ext/llama/api.cpp
DST="$B/build/overlay-o3"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }
[ -d "$B/build/overlay-o2" ] || { echo "FAILED: no overlay-o2 to build on" >&2; exit 2; }

rm -rf "$DST"
cp -r "$B/build/overlay-o2" "$DST"
mkdir -p "$DST/ext/llama"
apptainer exec "$SIF" cat "$SRC" > "$DST/ext/llama/api.cpp" || {
    echo "FAILED: could not read $SRC out of the image" >&2; exit 2; }
echo "extracted api.cpp: $(wc -l < "$DST/ext/llama/api.cpp") lines"

python3 - "$DST/ext/llama/api.cpp" <<'PYEOF'
import re, sys

path = sys.argv[1]
lines = open(path).read().split("\n")

# (signature prefix, the locals it must release) -- both read in full before this was written.
TARGETS = [
    ("void SlothGelu(", ["y", "d", "rp", "abs", "r"]),
    ("void SlothARS(",  ["z"]),
]

def extent(sig):
    starts = [i for i, l in enumerate(lines) if l.startswith(sig)]
    if len(starts) != 1:
        sys.exit("FAILED: %r matches %d definitions, expected exactly 1" % (sig, len(starts)))
    s = starts[0]
    for j in range(s + 1, len(lines)):
        if lines[j].startswith("}"):
            return s, j
    sys.exit("FAILED: no closing brace in column 0 after %r" % sig)

edits = []
for sig, names in TARGETS:
    s, e = extent(sig)
    body = "\n".join(lines[s:e + 1])
    # Premise of the step: these functions release nothing today; verify against the file.
    if "delete" in body:
        sys.exit("FAILED: %s already contains a delete; the leak this step attacks is not there "
                 "and the step's arithmetic no longer holds" % sig)
    for n in names:
        if not re.search(r"\bGroupElement \*%s = new GroupElement\[size\];" % re.escape(n), body):
            sys.exit("FAILED: %s does not allocate %r the way this patch expects" % (sig, n))
    edits.append((e, names, sig))

# Applied back to front so earlier insertion points keep their line numbers.
for e, names, sig in sorted(edits, reverse=True):
    ins = ["", "    // Released here, and only here, because every one of these is function-local:",
           "    // filled, read, and its result written into the caller's buffer before this",
           "    // returns. 36 of the 46 allocating functions in this file already do exactly this;",
           "    // these two were among the four that did not, and twelve layers' worth accumulated.",
           ] + ["    delete[] %s;" % n for n in names]
    lines[e:e] = ins
    print("patched %s: %s" % (sig.strip(), ", ".join("delete[] " + n for n in names)))

open(path, "w").write("\n".join(lines))
PYEOF
[ $? -eq 0 ] || { echo "FAILED: the patch refused" >&2; exit 3; }

# Read back from the file so the check is against the file, not the script's belief about it.
N=$(grep -c "delete\[\] \(y\|d\|rp\|abs\|r\|z\);" "$DST/ext/llama/api.cpp")
[ "$N" -ge 6 ] || { echo "FAILED: expected at least 6 new deletes, found $N" >&2; exit 3; }
echo "overlay-o3 ready: $(find "$DST" -type f | wc -l) files, $N releases in api.cpp"
find "$DST" -type f | sed "s#$DST/#  #"
