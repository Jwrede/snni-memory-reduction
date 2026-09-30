#!/bin/bash
# ARION in-run census hook, called by poll_peak.sh (SNNI_CENSUS_HOOK) at a material new maximum.
#
#   census_hook.sh <pid> <ts_ms> <rss_kb>        env: SNNI_CENSUS_DIR, SNNI_HEAPDUMP_PATH, CAMP
#
# Decomposition taken inside the measured run near its peak. Order: STOP -> pagemap -> USR1 queued ->
# CONT (the Go handler runs at once; WriteHeapDump stops the world for the write).
# Only the latest dump/pagemap pair is kept (a dump is heap-sized, 156 GB at a5).
set -uo pipefail
PID="$1"; TS="$2"; RSS="$3"
D="${SNNI_CENSUS_DIR:?}"; DUMP="${SNNI_HEAPDUMP_PATH:?}"; ERR="${SNNI_CENSUS_STDERR:?}"
mkdir -p "$D"
echo "=== census at ts=$TS rss_kb=$RSS ($(date -u +%T))"

kill -0 "$PID" 2>/dev/null || { echo "process gone"; exit 0; }
n_before=$(grep -a -c 'SNNI_HEAPDUMP|written=' "$ERR" 2>/dev/null || true); n_before=${n_before:-0}

rm -f "$D/pagemap.bin.new" "$DUMP".*
kill -STOP "$PID"
for _ in $(seq 1 200); do
    st=$(awk '{print $3}' "/proc/$PID/stat" 2>/dev/null)
    case "$st" in T|t|"") break ;; esac
    sleep 0.005
done
awk '/VmRSS/{print $2}' "/proc/$PID/status" > "$D/rss_at_snapshot_kb.new"
awk '/VmHWM/{print $2}' "/proc/$PID/status" > "$D/hwm_at_snapshot_kb.new"
python3 "$CAMP/attrib_go/pagemap_snapshot.py" "$PID" "$D/pagemap.bin.new" 2>&1
if [ ! -s "$D/pagemap.bin.new" ]; then
    echo "FAILED: no pagemap snapshot; not requesting a dump"
    kill -CONT "$PID"; exit 0
fi
kill -USR1 "$PID"
kill -CONT "$PID"

# Wait for the adapter to report the write (bounded; a dump of the largest heap here, ~570 GB at
# a0, takes minutes).
for _ in $(seq 1 3600); do
    n=$(grep -a -c 'SNNI_HEAPDUMP|written=' "$ERR" 2>/dev/null || true); n=${n:-0}
    [ "$n" -gt "$n_before" ] && break
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
done
f=$(ls -t "$DUMP".* 2>/dev/null | head -1)
if [ -z "$f" ] || [ "${n:-0}" -le "$n_before" ]; then
    echo "FAILED: no dump written after the signal"; rm -f "$D/pagemap.bin.new"; exit 0
fi
# Keep exactly one pair, named for the capture that produced it.
rm -f "$D/heap.dump" "$D/pagemap.bin"
mv "$f" "$D/heap.dump"
mv "$D/pagemap.bin.new" "$D/pagemap.bin"
mv "$D/rss_at_snapshot_kb.new" "$D/rss_at_snapshot_kb.txt"
mv "$D/hwm_at_snapshot_kb.new" "$D/hwm_at_snapshot_kb.txt"
echo "$TS $RSS" > "$D/CENSUS_AT"
echo "census ok: $(stat -c %s "$D/heap.dump") bytes, rss_at_snapshot=$(cat "$D/rss_at_snapshot_kb.txt")"
