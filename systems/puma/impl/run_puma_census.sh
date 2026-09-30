#!/bin/bash
# PUMA CENSUS RUNNER (diagnostic; generated from run_puma.sh 2026-08-04, one block changed).
#
# Nodes started one by one (not `nodectl up`) so each pid is known; the peak is per process, all
# six polled and the binding one published (written to published_party.txt by the derive).
set -uo pipefail

RESULTS="${PUMA_RESULTS:?need PUMA_RESULTS}"
CONFIG="${PUMA_CONFIG:-/opt/puma/examples/python/ml/flax_bert/3pc.json}"
DRIVER="${PUMA_DRIVER:?need PUMA_DRIVER}"
export PUMA_GATE_FILE="$RESULTS/gate.txt"
# CENSUS VARIANT (p_census, diagnostic): /opt/census sitecustomize arms a gc census of arrays at
# each new RSS max in every cluster process; guarded by the path binding and SNNI_PUMA_CENSUS=1.
export PYTHONPATH=/opt/census:/opt/puma
export SNNI_PUMA_CENSUS=1
export SNNI_PUMA_CENSUS_DIR="$RESULTS"
export PYTHONUNBUFFERED=1

mkdir -p "$RESULTS"
rm -f "$RESULTS"/node*.pid "$RESULTS"/driver.pid "$RESULTS/gate.txt"

md5sum "$DRIVER" | tee "$RESULTS/driver_md5.txt"
python3 -c "import spu; print('spu', spu.__version__)" 2>/dev/null | tee "$RESULTS/stack_versions.txt"

cd /opt/puma || exit 2

NODES=$(python3 -c "
import json,sys
print(' '.join(json.load(open('$CONFIG'))['nodes'].keys()))
")
[ -n "$NODES" ] || { echo "FAILED: no nodes in $CONFIG" >&2; exit 2; }

echo "=== starting nodes: $NODES"
i=0
for n in $NODES; do
    python3 examples/python/utils/nodectl.py --config "$CONFIG" start -n "$n" \
        > "$RESULTS/node${i}_stdout.log" 2> "$RESULTS/node${i}_stderr.log" &
    echo $! > "$RESULTS/node${i}.pid"
    echo "  $n pid=$(cat "$RESULTS/node${i}.pid")"
    i=$((i + 1))
done

# Wait for the node grpc ports to listen rather than sleeping a guessed interval.
PORTS=$(python3 -c "
import json
for v in json.load(open('$CONFIG'))['nodes'].values():
    print(v.split(':')[-1])
")
for _ in $(seq 1 600); do
    up=0
    for p in $PORTS; do
        (echo > /dev/tcp/127.0.0.1/"$p") 2>/dev/null && up=$((up + 1))
    done
    [ "$up" = "$(echo "$PORTS" | wc -w)" ] && break
    sleep 0.5
done
echo "=== $up of $(echo "$PORTS" | wc -w) node ports listening"
if [ "$up" != "$(echo "$PORTS" | wc -w)" ]; then
    echo "FAILED: not every node bound its port; this run has no cluster" >&2
    for f in "$RESULTS"/node*_stderr.log; do echo "--- $f"; tail -5 "$f"; done
    exit 3
fi

echo "=== driver: $DRIVER"
T0=$(date +%s)
python3 "$DRIVER" --config "$CONFIG" \
    > "$RESULTS/driver_stdout.log" 2> "$RESULTS/driver_markers.txt" &
DRV=$!
echo "$DRV" > "$RESULTS/driver.pid"

RC=0; wait "$DRV" || RC=$?
echo "=== driver rc=$RC wall=$(( $(date +%s) - T0 ))s"
echo "$RC" > "$RESULTS/program_exit_code.txt"

# Stop the cluster. The nodes never exit on their own: they are services.
for f in "$RESULTS"/node*.pid; do
    [ -r "$f" ] || continue
    kill "$(cat "$f")" 2>/dev/null
done
sleep 2
for f in "$RESULTS"/node*.pid; do
    [ -r "$f" ] || continue
    kill -9 "$(cat "$f")" 2>/dev/null
done

# The markers the harness requires; each is checked separately with a named failure.
grep -h '^SEEDED|'   "$RESULTS/driver_markers.txt" > "$RESULTS/seeded.txt"   2>/dev/null
grep -h '^WORKLOAD|' "$RESULTS/driver_markers.txt" > "$RESULTS/workload.txt" 2>/dev/null
if [ ! -s "$RESULTS/seeded.txt" ]; then
    echo "FAILED: no SEEDED| marker; the pins did not take" >&2
    [ "$RC" = 0 ] && RC=5
fi
if [ ! -s "$RESULTS/workload.txt" ]; then
    echo "FAILED: no WORKLOAD| marker; what ran is not recorded" >&2
    [ "$RC" = 0 ] && RC=5
fi
if grep -q '^GATE_DEGENERATE|' "$RESULTS/driver_markers.txt" 2>/dev/null; then
    echo "FAILED: the gate observable is degenerate (all zero)" >&2
    [ "$RC" = 0 ] && RC=6
fi

NG=$(wc -l < "$RESULTS/gate.txt" 2>/dev/null || echo 0)
EXPECTED=$(( ${PUMA_SEQ_LEN:-128} * 768 ))
echo "gate values: $NG (expected $EXPECTED)"
if [ "$NG" != "$EXPECTED" ]; then
    echo "FAILED: gate observable missing or short ($NG of $EXPECTED)" >&2
    [ "$RC" = 0 ] && RC=5
fi

exit "$RC"
