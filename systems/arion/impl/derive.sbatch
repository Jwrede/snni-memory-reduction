#!/bin/bash
#SBATCH --job-name=arionderive
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/arion/derive-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/arion/derive-%j.out
#
#   sbatch --export=ALL,STEP=a0_default,RUN=1 derive.sbatch
#
# Expected output: an empty heap table (Go bypasses malloc; LD_PRELOAD sees nothing, ONBOARDING.md).
# Produced: smaps half (file-backed + anonymous mappings) and the Go collector's live-after-GC
# series, published beside the peak as evidence.
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/arion
R=$B/results/$STEP${RUN:+/run$RUN}
SIF="${SIF:-$B/sif/arion-$STEP.sif}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 1; }
[ -d "$R" ] || { echo "FAILED: no results at $R"; exit 1; }

FOUND=0
for P in "$R"/poll*; do
    [ -d "$P" ] || continue
    for f in "$P/peaks/pm.maps"; do
        [ -s "$f" ] || { echo "FAILED: missing or empty $f"; exit 1; }
    done
    FOUND=$((FOUND + 1))

    echo "=== scan cost actually paid (ts rss scan_ms)"
    awk '{s+=$3; n++; if ($3>m) m=$3}
         END {if (n) printf "samples=%d  mean=%.0f ms  max=%.0f ms\n", n, s/n, m}' \
        "$P/peaks/PM_INDEX" 2>/dev/null || echo "(no PM_INDEX)"

    # The load-base rewrite (see other systems' derives): corrects the shared symboliser's
    # one-page error where p_vaddr != p_offset, without touching the shared tool.
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

    # Count rows, don't just test non-empty: the scan always writes header lines, so `[ -s ]` is
    # true even with no recorded allocations, which is the finding on this system, not a tool error.
    NSITES=$(awk '!/^#/{n++} END{print n+0}' "$P/peaks/resident.txt" 2>/dev/null)
    TOT=$(awk '!/^#/{s+=$1} END{print s+0}' "$P/peaks/resident.txt" 2>/dev/null)
    echo "=== recorder table: ${NSITES:-0} call sites, ${TOT:-0} resident bytes"
    if [ "${NSITES:-0}" -gt 0 ]; then
        python3 "$C/attrib_resident/resolve_resident.py" \
            "$P/peaks/resident.txt" "$P/peaks/pm.maps.vaddr" "$R/objects_resident.jsonl" \
            --exec "apptainer exec $SIF" --top 15 || exit 1
    else
        : > "$R/objects_resident.jsonl"
        echo "=== no call sites to name; the heap half of the decomposition does not exist here"
    fi

    if [ "${NSITES:-0}" -eq 0 ] || [ "${TOT:-0}" -lt 1048576 ]; then
        # Named, not implied, so a downstream reader can tell "no attribution mechanism" apart
        # from "capture truncated" -- both would otherwise produce the same empty file.
        echo "GO_RUNTIME_NO_MALLOC_ATTRIBUTION: the recorder saw ${TOT:-0} resident bytes across"
        echo "  ${NSITES:-0} call sites. Go's runtime obtains memory by mmap and manages it itself,"
        echo "  so an allocator interceptor is blind here BY CONSTRUCTION. No step of this system"
        echo "  can be object-conformant until a Go-side mechanism exists; the peak is still valid."
        echo "GO_RUNTIME_NO_MALLOC_ATTRIBUTION" > "$R/ATTRIBUTION_MISSING.txt"
    fi

    echo "=== assembling what the smaps half can say"
    # Heap side comes from the `_goheap` channel where available, since the malloc recorder is
    # blind to Go's allocations and the peak would otherwise land entirely in `anon:residual`.
    # One run per step: the in-run census writes its table into the run itself. The separate
    # `_goheap` channel is the retired two-run arrangement and is read only if no in-run table exists.
    GOTBL="$R/objects_goheap.jsonl"
    [ -s "$GOTBL" ] || GOTBL="$B/results/${STEP}_goheap${RUN:+/run$RUN}/objects_goheap.jsonl"
    if [ -s "$GOTBL" ]; then
        echo "heap side: the Go census channel ($GOTBL)"
        HEAPSRC="$GOTBL"
    else
        echo "heap side: the malloc recorder -- on this system that names nothing, so the peak will"
        echo "           come out as anon:residual and no step can be chosen from it"
        HEAPSRC="$R/objects_resident.jsonl"
    fi
    python3 "$C/smaps_objects.py" "$P/peaks" "$R/objects_host.jsonl" \
        --heap "$HEAPSRC" || exit 1
done

[ "$FOUND" -gt 0 ] || { echo "FAILED: no poll directory under $R"; exit 1; }

# The Go collector's own view, from the run's own markers. Evidence, not the metric.
if [ -s "$R/markers.txt" ]; then
    grep -oE '[0-9]+->[0-9]+->[0-9]+ MB' "$R/markers.txt" \
        | awk -F'->' '{split($3,a," "); print a[1]+0}' > "$R/gctrace_live_mb.txt"
    awk '{if ($1>m) m=$1; n++} END{printf "gc_cycles=%d  max_live_after_gc_mb=%d\n", n, m}' \
        "$R/gctrace_live_mb.txt" | tee "$R/gctrace_summary.txt"
fi

echo "DERIVE_OK"
