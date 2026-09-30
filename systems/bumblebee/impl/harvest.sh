#!/bin/bash
# Harvests one BumbleBee step from the cluster and regenerates the tables (local).
#
#   harvest.sh <step_id>
#
# Wrapper over tools/harvest_step.sh; also copies replicate 1's RUN_META to the step-level
# logs/RUN_META (make_steps.py reads impl_fingerprint there).
set -uo pipefail

cd "$(dirname "$0")/../../.." || exit 1
STEP="${1:?usage: harvest.sh <step_id>}"
SYS=systems/bumblebee
LOGROOT=/scratch/tmp/j_wred02/snni_campaign/bumblebee/results

bash tools/harvest_step.sh "$SYS" "$STEP" palma "$LOGROOT" || exit 1

SD="$SYS/steps/$STEP"
for r in 1 2 3; do
    if [ -r "$SD/logs/run$r/RUN_META" ]; then
        cp "$SD/logs/run$r/RUN_META" "$SD/logs/RUN_META"
        break
    fi
done

echo "=== what landed"
for r in "$SD"/logs/run*; do
    [ -d "$r" ] || continue
    hwm=$(awk -F= '/^# hwm_kb=/{print $2}' "$r"/poll_*/vmrss.log 2>/dev/null | sort -n | tail -1)
    frz=$(awk -F= '/^# frozen_ms=/{print $2}' "$r"/poll_*/vmrss.log 2>/dev/null | sort -n | tail -1)
    g=$(wc -l < "$r/gate.txt" 2>/dev/null || echo 0)
    echo "  $(basename "$r"): VmHWM=${hwm:-none} kB  frozen=${frz:-none} ms  gate_values=$g"
done

echo "=== drops must be 0, or the ranking lost allocations to table overflow"
grep -h '^# live_allocations' "$SD"/logs/run*/poll_*/peaks/resident.txt 2>/dev/null | sort -u

echo "=== regenerating"
python3 tools/make_steps.py "$SYS" || exit 1
python3 tools/make_steps.py "$SYS" --check || exit 1
echo "HARVEST_OK"
