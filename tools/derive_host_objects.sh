#!/bin/bash
# Derives one step's canonical host decomposition. Runs on the measurement host (addr2line needs
# the run's image), after the sweep.
#
#   derive_host_objects.sh <clean-results-dir> <pmrec-results-dir> <image> <out.jsonl>
#
# Libraries: clean run's peak smaps. Objects: pmrec run's resident table. Remainder: anon:residual.
set -uo pipefail
CLEAN="${1:?need clean results dir}"
PMREC="${2:?need pmrec results dir}"
IMAGE="${3:?need step image}"
OUT="${4:?need output jsonl}"
# Optional 5th arg: extra apptainer binds, for a system that runs a binary NOT in its image.
# SIGMA-GPU binds the measured sigma binary into the container, so resolving without that bind hands
# addr2line the image's stock program (a 32.0 GiB "near SigmaPeer::wait()", which allocates nothing;
# against the real binary the same offset is cpuMalloc at gpu_mem.cu:77). Empty by default.
EXTRA_BIND="${5:-}"
TOOLS="$(cd "$(dirname "$0")" && pwd)"

peaks_of() {  # the ONE poller's peaks dir under the given directory
    # A multi-party run has one poll_<pid>/peaks per party, and taking whichever `find` lists
    # first decomposes an arbitrary party. So exactly one peaks/ must be found: hand this script
    # the party's own poll_<pid> directory (tools/derive_party.sbatch does, per explicit pid).
    local d n
    d=$(find "$1" -type d -name peaks 2>/dev/null)
    n=$(printf '%s\n' "$d" | grep -c .)
    [ "$n" -ge 1 ] || { echo "ERROR: no peaks/ under $1" >&2; return 1; }
    [ "$n" -eq 1 ] || { echo "ERROR: $n peaks/ directories under $1; pass one party's poll_<pid> dir" >&2; return 1; }
    echo "$d"
}

CP=$(peaks_of "$CLEAN") || exit 1
PP=$(peaks_of "$PMREC") || exit 1
[ -s "$PP/resident.txt" ] || { echo "ERROR: $PP/resident.txt missing or empty" >&2; exit 1; }
[ -s "$PP/pm.maps" ]     || { echo "ERROR: $PP/pm.maps missing; addresses are unresolvable" >&2; exit 1; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# The symboliser runs in the image via apptainer (this once said `podman run`, which on PALMA left
# every addr2line failing silently, since the resolver falls back to <basename>+0x<offset>): MOAI-GPU
# ranked test+0xf8fae/test+0xd71ab at 84% of the peak, which inside the image resolve to
# ComplexRoots (fft.cu:12) and _M_default_append. No measurement changes, only what objects are called.
python3 "$TOOLS/attrib_resident/resolve_resident.py" \
    "$PP/resident.txt" "$PP/pm.maps" "$TMP/objects_resident.jsonl" \
    --exec "apptainer exec --bind /scratch/tmp/j_wred02:/scratch/tmp/j_wred02 $EXTRA_BIND $IMAGE" \
    --top 20 || exit 1

python3 "$TOOLS/smaps_objects.py" "$CP" "$OUT" --heap "$TMP/objects_resident.jsonl" || exit 1
cp "$TMP/objects_resident.jsonl" "$(dirname "$OUT")/objects_resident.jsonl"
echo "wrote $OUT and $(dirname "$OUT")/objects_resident.jsonl"
