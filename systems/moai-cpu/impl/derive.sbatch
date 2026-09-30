#!/bin/bash
#SBATCH --job-name=moaicpuderive
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/derive-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/derive-%j.out
#
#   sbatch --export=ALL,STEP=s0_default,RUN=1 derive.sbatch
#
# One replicate's raw run -> canonical object tables (cluster; addr2line needs the step's image).
# Tables older than the SEAL aligned_alloc hook (2026-07-30) are invalid.
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/moai-cpu
R=$B/results/$STEP${RUN:+/run$RUN}
# The image comes from the RUN that was measured, not from the step's name: a probe step's binary
# may have fallen back to the baseline image, which the step name alone can't say. Reads what the
# run recorded (RUN_META) rather than re-deriving the guess, falling back only when that's absent.
if [ -z "${SIF:-}" ] && [ -r "$R/RUN_META" ]; then
    SIF=$(grep -aoE 'image=[^[:space:]]+' "$R/RUN_META" 2>/dev/null | head -1 | cut -d= -f2-)
    [ -r "${SIF:-}" ] || SIF=""
fi
SIF="${SIF:-$B/sif/moai-cpu-$STEP.sif}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 1; }
[ -d "$R" ] || { echo "FAILED: no results at $R"; exit 1; }

BEST_DIR=""
BEST_HWM=0
FOUND=0

for P in "$R"/poll*; do
    [ -d "$P" ] || continue
    TAG=$(basename "$P" | sed 's/^poll_*//')
    [ -n "$TAG" ] || TAG=single
    for f in "$P/peaks/resident.txt" "$P/peaks/pm.maps"; do
        [ -s "$f" ] || { echo "FAILED: missing or empty $f"; exit 1; }
    done
    FOUND=$((FOUND + 1))

    echo "=== process $TAG: scan cost actually paid (ts rss scan_ms)"
    awk '{s+=$3; n++; if ($3>m) m=$3}
         END {if (n) printf "samples=%d  mean=%.0f ms  max=%.0f ms\n", n, s/n, m}' \
        "$P/peaks/PM_INDEX" 2>/dev/null || echo "(no PM_INDEX)"

    # The shared symboliser's offset formula is only exact when p_vaddr==p_offset; rewrites this
    # run's own maps snapshot to be load-base relative rather than touching the shared tool.
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

    echo "=== process $TAG: naming the call sites"
    # `--program-root` names a row by what a frame IS rather than its fixed index (call-chain depth
    # varies by step); safe unconditionally since a pre-C3 table just keeps its old name.
    python3 "$C/attrib_resident/resolve_resident.py" \
        "$P/peaks/resident.txt" "$P/peaks/pm.maps.vaddr" "$R/objects_resident_$TAG.jsonl" \
        --exec "apptainer exec $SIF" --top 15 --program-root /root/moai || exit 1

    # Published host table comes from the pool recorder, not pmrec (whose frees never fire since
    # SEAL intercepts them); pmrec's table is kept separately as objects_host_pmrec.jsonl.
    echo "=== process $TAG: assembling the host decomposition"
    python3 "$C/smaps_objects.py" "$P/peaks" "$R/objects_host_pmrec_$TAG.jsonl" \
        --heap "$R/objects_resident_$TAG.jsonl" || exit 1
    if [ -s "$P/peaks/resident_pool.txt" ]; then
        MAPS2="$P/peaks/pool.maps"; [ -r "$MAPS2" ] || MAPS2="$P/peaks/pm.maps.vaddr"
        python3 "$C/attrib_resident/resolve_resident.py" \
            "$P/peaks/resident_pool.txt" "$MAPS2" "$R/objects_pool_$TAG.jsonl" \
            --exec "apptainer exec $SIF" --top 20 --program-root /root/moai || exit 1
        # `--retention-row`: names the 42% no live pool object covers via pmrec instead of `anon:`;
        # stays a retention aggregate, admitted only under `runtime_knob: yes`.
        python3 "$C/smaps_objects.py" "$P/peaks" "$R/objects_host_$TAG.jsonl" \
            --heap "$R/objects_pool_$TAG.jsonl" --retention-row sealpool:retained || exit 1
    else
        echo "NOTE: no pool table in this run; the published host table falls back to pmrec's"
        cp "$R/objects_host_pmrec_$TAG.jsonl" "$R/objects_host_$TAG.jsonl" || exit 1
    fi

    # High-water mark: falls back to max(column 4) when the poller died before its summary line;
    # refuses a trace with neither rather than ranking that party at zero.
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
    echo "process $TAG VmHWM = $HWM kB"
    if [ "$HWM" -gt "$BEST_HWM" ]; then
        BEST_HWM="$HWM"
        BEST_DIR="$TAG"
    fi
done

[ "$FOUND" -gt 0 ] || { echo "FAILED: no poll directory (poll or poll_<pid>) under $R"; exit 1; }
[ -n "$BEST_DIR" ] || { echo "FAILED: no process carried a VmHWM"; exit 1; }

cp "$R/objects_host_$BEST_DIR.jsonl"     "$R/objects_host.jsonl"     || exit 1
cp "$R/objects_resident_$BEST_DIR.jsonl" "$R/objects_resident.jsonl" || exit 1
echo "published process = $BEST_DIR (VmHWM $BEST_HWM kB)" | tee "$R/published_party.txt"

# Which process bound, and by how much; recorded rather than inferred later from file size.
{
    echo "# process VmHWM_kB"
    for P in "$R"/poll*; do
        [ -d "$P" ] || continue
        t=$(basename "$P" | sed 's/^poll_*//')
        h=$(awk '/^# hwm_kb=/{split($0,a,"="); if (a[2]+0 > m) m = a[2]+0}
                 !/^#/ && NF >= 4 && $4 ~ /^[0-9]+$/ {if ($4+0 > m) m = $4+0}
                 END{print m+0}' "$P/vmrss.log" 2>/dev/null)
        echo "$t ${h:-0}"
    done
} | sort -k2 -rn | tee "$R/party_peaks.txt"

echo "DERIVE_OK"
