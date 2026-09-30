#!/bin/bash
#SBATCH --job-name=arion-parsechk
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
# 64G after an OOM at 8G (job 45857434: addr -> size held for every object; fixed). Sample count unknown.
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/arion/parsechk-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/arion/parsechk-%j.out

# Tests parse_heapdump.py on a real Go heap dump from a killed run (no paired pagemap): record framing
# and MemProfileRate coverage only. Nothing measured or published.
set -uo pipefail

CAMP=/scratch/tmp/j_wred02/snni_campaign
DUMP="$CAMP/arion/results/a0_default_goheap/run1/heap.dump.1"
OUT="$CAMP/arion/results/a0_default_goheap/run1/objects_go_PARSECHECK.jsonl"

[ -s "$DUMP" ] || { echo "FAILED: no dump at $DUMP" >&2; exit 2; }
[ -r "$CAMP/attrib_go/parse_heapdump.py" ] || {
    echo "FAILED: parse_heapdump.py is not deployed -- the very defect this check follows" >&2
    exit 2; }

echo "dump: $(stat -c %s "$DUMP") bytes"
T0=$(date +%s)
python3 "$CAMP/attrib_go/parse_heapdump.py" "$DUMP" "$OUT"
RC=$?
echo "parse rc=$RC after $(( $(date +%s) - T0 ))s"
[ "$RC" -eq 0 ] || { echo "PARSER FAILED on a real dump" >&2; exit 3; }

echo "=== what came out ==="
echo "objects: $(wc -l < "$OUT")"
echo "with a site vs unsampled:"
python3 - "$OUT" <<'PYEOF'
import json, sys
from collections import Counter
n = 0
sized = Counter()
counted = Counter()
for line in open(sys.argv[1]):
    o = json.loads(line)
    n += 1
    key = "unsampled" if o.get("site") == "unsampled" else "sited"
    counted[key] += 1
    sized[key] += o.get("size", 0)
tot = sum(sized.values()) or 1
for k in ("sited", "unsampled"):
    print("  %-10s %9d objects  %14d bytes  %5.1f%% of bytes"
          % (k, counted[k], sized[k], 100.0 * sized[k] / tot))
print("  total      %9d objects  %14d bytes" % (n, sum(sized.values())))
PYEOF
echo "=== the ten largest sites by allocated bytes (NOT resident, NOT publishable) ==="
python3 - "$OUT" <<'PYEOF'
import json, sys
from collections import Counter
by = Counter()
for line in open(sys.argv[1]):
    o = json.loads(line)
    by[o.get("site", "?")] += o.get("size", 0)
for site, b in by.most_common(10):
    print("  %14d  %s" % (b, site[:90]))
PYEOF
gzip -f "$OUT" 2>/dev/null || true
echo "PARSECHECK_OK"
