#!/bin/bash
#SBATCH --job-name=bbderive
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/bumblebee/derive-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/bumblebee/derive-%j.out
#
#   sbatch --export=ALL,STEP=b0_default,RUN=1 derive.sbatch
#
# One replicate's raw run -> canonical object tables. On the cluster (addr2line needs the run's image).
# Both parties derived separately; published table = party with the higher VmHWM.
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/bumblebee
R=$B/results/$STEP${RUN:+/run$RUN}
# The image is read from the run's RUN_META, not derived from the step name: b5-b8 and geometry
# probes resolve to no file if the step name is taken literally.
if [ -z "${SIF:-}" ] && [ -r "$R/RUN_META" ]; then
    SIF=$(grep -aoE 'image=[^[:space:]]+' "$R/RUN_META" 2>/dev/null | head -1 | cut -d= -f2-)
    [ -r "${SIF:-}" ] || SIF=""
fi
SIF="${SIF:-$B/sif/bumblebee-$STEP.sif}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 1; }

BEST_DIR=""
BEST_HWM=0
FOUND=0

for P in "$R"/poll_*; do
    [ -d "$P" ] || continue
    TAG=$(basename "$P" | sed 's/^poll_*//')
    [ -n "$TAG" ] || TAG=single
    for f in "$P/peaks/resident.txt" "$P/peaks/pm.maps"; do
        [ -s "$f" ] || { echo "FAILED: missing or empty $f"; exit 1; }
    done
    FOUND=$((FOUND + 1))

    echo "=== party $TAG: scan cost actually paid (ts rss scan_ms)"
    awk '{s+=$3; n++; if ($3>m) m=$3}
         END {if (n) printf "samples=%d  mean=%.0f ms  max=%.0f ms\n", n, s/n, m}' \
        "$P/peaks/PM_INDEX" 2>/dev/null || echo "(no PM_INDEX)"

    # Works around an off-by-one-page in the shared symboliser locally, without touching it: this
    # binary's text segment has p_vaddr != p_offset, so rewrites this run's maps snapshot to make
    # the file-offset column load-base-relative (exact for addr2line); reported upstream separately.
    python3 - "$P/peaks/pm.maps" "$P/peaks/pm.maps.vaddr" <<'PYEOF' || exit 1
import sys
src, dst = sys.argv[1], sys.argv[2]
rows = []
base = {}
for ln in open(src, errors="replace"):
    p = ln.split()
    if len(p) >= 6 and "-" in p[0] and p[5].startswith("/"):
        lo = int(p[0].split("-")[0], 16)
        base[p[5]] = min(base.get(p[5], lo), lo)
    rows.append(ln)
out = open(dst, "w")
fixed = 0
for ln in rows:
    p = ln.split()
    if len(p) >= 6 and "-" in p[0] and p[5] in base:
        lo = int(p[0].split("-")[0], 16)
        p[2] = "%08x" % (lo - base[p[5]])
        out.write(" ".join(p) + "\n")
        fixed += 1
    else:
        out.write(ln)
out.close()
print("rewrote %d file-backed mapping offsets to be load-base relative" % fixed)
PYEOF

    echo "=== party $TAG: naming the call sites"
    python3 "$C/attrib_resident/resolve_resident.py" \
        "$P/peaks/resident.txt" "$P/peaks/pm.maps.vaddr" "$R/objects_resident_$TAG.jsonl" \
        --exec "apptainer exec $SIF" --top 15 || exit 1

    echo "=== party $TAG: assembling the host decomposition"
    python3 "$C/smaps_objects.py" "$P/peaks" "$R/objects_host_$TAG.jsonl" \
        --heap "$R/objects_resident_$TAG.jsonl" || exit 1

    HWM=$(awk '/^# hwm_kb=/{split($0,a,"="); print a[2]}' "$P/vmrss.log" 2>/dev/null)
    [ -n "$HWM" ] || HWM=0
    echo "party $TAG VmHWM = $HWM kB"
    if [ "$HWM" -gt "$BEST_HWM" ]; then
        BEST_HWM="$HWM"
        BEST_DIR="$TAG"
    fi
done

[ "$FOUND" -gt 0 ] || { echo "FAILED: no poll_* directory under $R"; exit 1; }
[ -n "$BEST_DIR" ] || { echo "FAILED: no party carried a VmHWM"; exit 1; }

# The published decomposition is the binding party's (both must fit, so the larger peak matters).
cp "$R/objects_host_$BEST_DIR.jsonl"     "$R/objects_host.jsonl"     || exit 1
cp "$R/objects_resident_$BEST_DIR.jsonl" "$R/objects_resident.jsonl" || exit 1
echo "published party = $BEST_DIR (VmHWM $BEST_HWM kB)" | tee "$R/published_party.txt"
echo "DERIVE_OK"
