#!/bin/bash
#SBATCH --job-name=boltderive
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/bolt/derive-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/bolt/derive-%j.out
#
#   sbatch --export=ALL,STEP=s0_default,RUN=1 derive.sbatch
#
# One replicate's raw run -> canonical object tables. On the cluster (addr2line needs the run's image).
# ALICE and BOB derived separately; published decomposition = party with the higher VmHWM.
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/bolt
R=$B/results/$STEP${RUN:+/run$RUN}
# The image comes from the run's own RUN_META (what was actually measured), not guessed from the
# step's name, since a probe may have fallen back to the baseline image; falls back to the old
# name-based guess only for runs predating snni_run_meta.
if [ -z "${SIF:-}" ] && [ -r "$R/RUN_META" ]; then
    SIF=$(grep -aoE 'image=[^[:space:]]+' "$R/RUN_META" 2>/dev/null | head -1 | cut -d= -f2-)
    [ -r "${SIF:-}" ] || SIF=""
fi
SIF="${SIF:-$B/sif/bolt-$STEP.sif}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 1; }
[ -d "$R" ] || { echo "FAILED: no results at $R"; exit 1; }

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

    # Rewrites this run's maps snapshot so file-offset becomes load-base-relative (the shared
    # symboliser's offset formula is only exact when p_vaddr==p_offset); the shared tool itself
    # is left alone since changing it would move every system's published tables at once.
    python3 - "$P/peaks/pm.maps" "$P/peaks/pm.maps.vaddr" <<'PYEOF' || exit 1
import sys
src, dst = sys.argv[1], sys.argv[2]
rows, base = [], {}
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

    # High-water mark taken from the summary line (`# hwm_kb=`) or, if the poller was killed before
    # writing it, from the max of column 4 (VmHWM) across samples; refuses a trace with neither
    # rather than silently ranking that party at zero.
    HWM=$(awk '/^# hwm_kb=/{split($0,a,"="); if (a[2]+0 > m) m = a[2]+0}
               !/^#/ && NF >= 4 && $4 ~ /^[0-9]+$/ {if ($4+0 > m) m = $4+0}
               END{print m+0}' "$P/vmrss.log" 2>/dev/null)
    [ -n "$HWM" ] || HWM=0
    if [ "$HWM" = 0 ]; then
        echo "FAILED: $P/vmrss.log carries no VmHWM at all, in the summary or in the samples."
        echo "        A trace with no high-water mark cannot rank its process, and ranking it as"
        echo "        zero is how the wrong party's decomposition gets published."
        exit 1
    fi
    grep -q '^# hwm_kb=' "$P/vmrss.log" 2>/dev/null \
        || echo "NOTE: no summary line in $P/vmrss.log (poller killed before it finished); VmHWM taken from the samples"
    echo "party $TAG VmHWM = $HWM kB"
    if [ "$HWM" -gt "$BEST_HWM" ]; then
        BEST_HWM="$HWM"
        BEST_DIR="$TAG"
    fi
done

[ "$FOUND" -gt 0 ] || { echo "FAILED: no poll_* directory under $R"; exit 1; }
[ -n "$BEST_DIR" ] || { echo "FAILED: no party carried a VmHWM"; exit 1; }

cp "$R/objects_host_$BEST_DIR.jsonl"     "$R/objects_host.jsonl"     || exit 1
cp "$R/objects_resident_$BEST_DIR.jsonl" "$R/objects_resident.jsonl" || exit 1
echo "published party = $BEST_DIR (VmHWM $BEST_HWM kB)" | tee "$R/published_party.txt"

# Which party bound, and by how much, recorded rather than assumed (ALICE expected ~20x BOB).
{
    echo "# party VmHWM_kB"
    for P in "$R"/poll_*; do
        [ -d "$P" ] || continue
        t=$(basename "$P" | sed 's/^poll_*//')
        h=$(awk '/^# hwm_kb=/{split($0,a,"="); if (a[2]+0 > m) m = a[2]+0}
                 !/^#/ && NF >= 4 && $4 ~ /^[0-9]+$/ {if ($4+0 > m) m = $4+0}
                 END{print m+0}' "$P/vmrss.log" 2>/dev/null)
        echo "$t ${h:-0}"
    done
} | sort -k2 -rn | tee "$R/party_peaks.txt"

echo "DERIVE_OK"
