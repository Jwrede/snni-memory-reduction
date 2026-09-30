#!/usr/bin/env python3
"""Take the decomposition at a NAMED PHASE MARKER (SNNI_MARKER_FILE), not at the RSS high-water: on MOAI-CPU the whole second layer sits within 0.122% of the peak, so the high-water instant is undefined. No rebuild; peak record untouched.

    patch_poller_markersample.py <path to poll_peak.sh>
"""
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
P = sys.argv[1]
if not os.path.isfile(P):
    sys.exit(f"FAILED: {P} is not a file")

s = open(P, encoding="utf-8").read()
if "SNNI_MARKER_FILE" in s:
    print("  already present: marker sampling")
    sys.exit(0)

# 1. snapshot_peak takes an optional marker name. With one, it must not touch the peak record.
OLD1 = """snapshot_peak() {
    local ts="$1" rss="$2"
"""
NEW1 = """snapshot_peak() {
    local ts="$1" rss="$2" marker="${3:-}"
"""
if s.count(OLD1) != 1:
    sys.exit(f"FAILED: snapshot_peak's header matched {s.count(OLD1)} times, expected 1")
s = s.replace(OLD1, NEW1, 1)

# 2. The peak record belongs to the high-water path alone.
OLD2 = """    cp "/proc/$PID/smaps"  "$OUTDIR/peaks/.peak.smaps.tmp"  2>/dev/null &&
        mv "$OUTDIR/peaks/.peak.smaps.tmp"  "$OUTDIR/peaks/peak.smaps"
    cp "/proc/$PID/status" "$OUTDIR/peaks/.peak.status.tmp" 2>/dev/null &&
        mv "$OUTDIR/peaks/.peak.status.tmp" "$OUTDIR/peaks/peak.status"
    echo "$ts $rss" > "$OUTDIR/peaks/PEAK"
"""
NEW2 = """    # A MARKER SAMPLE IS NOT A PEAK. It records a declared instant; the peak record keeps its
    # existing meaning and is written only when RSS actually reached a new maximum.
    if [ -z "$marker" ]; then
        cp "/proc/$PID/smaps"  "$OUTDIR/peaks/.peak.smaps.tmp"  2>/dev/null &&
            mv "$OUTDIR/peaks/.peak.smaps.tmp"  "$OUTDIR/peaks/peak.smaps"
        cp "/proc/$PID/status" "$OUTDIR/peaks/.peak.status.tmp" 2>/dev/null &&
            mv "$OUTDIR/peaks/.peak.status.tmp" "$OUTDIR/peaks/peak.status"
        echo "$ts $rss" > "$OUTDIR/peaks/PEAK"
    else
        # Link, do not copy: the tables were written to a fresh inode by `mv`, so a link keeps
        # exactly this marker's bytes for the cost of one directory entry.
        ln -f "$OUTDIR/peaks/resident.txt" \\
              "$OUTDIR/peaks/marker_${marker}.txt" 2>/dev/null || true
        [ -s "$OUTDIR/peaks/resident_pool.txt" ] && \\
            ln -f "$OUTDIR/peaks/resident_pool.txt" \\
                  "$OUTDIR/peaks/marker_${marker}_pool.txt" 2>/dev/null || true
        echo "$ts $rss $marker" >> "$OUTDIR/peaks/MARKER_INDEX"
    fi
"""
if s.count(OLD2) != 1:
    sys.exit(f"FAILED: the peak-record block matched {s.count(OLD2)} times, expected 1")
s = s.replace(OLD2, NEW2, 1)

# 3. A marker sample ignores the duty gate: skipping it would link this marker's name to an older
#    sample, i.e. a plausible file with the wrong contents.
OLD2b = """    if [ -n "${SNNI_PM_TABLE:-}" ] && [ -x "${SNNI_PM_SAMPLE:-}" ] && {
           [ "$ts" -ge "$PM_NEXT_TS" ] || [ "$PM_LAST_RSS" -eq 0 ] ||"""
NEW2b = """    # A MARKER SAMPLE IGNORES THE DUTY GATE, and it has to. The gate exists to keep the poller
    # cheap during a rising ramp, but a marker asks for the composition AT A DECLARED INSTANT: if
    # the gate skipped it, the link below would name this marker and point at an older sample --
    # a file with a plausible name and the wrong contents, which is the failure shape this
    # campaign keeps meeting.
    if [ -n "${SNNI_PM_TABLE:-}" ] && [ -x "${SNNI_PM_SAMPLE:-}" ] && {
           [ -n "$marker" ] ||
           [ "$ts" -ge "$PM_NEXT_TS" ] || [ "$PM_LAST_RSS" -eq 0 ] ||"""
if s.count(OLD2b) != 1:
    sys.exit(f"FAILED: the duty gate matched {s.count(OLD2b)} times, expected 1")
s = s.replace(OLD2b, NEW2b, 1)

# 4. Follow the marker file, without a fork per poll.
OLD3 = """while kill -0 "$PID" 2>/dev/null; do
"""
NEW3 = """# THE MARKER FILE, held open so the loop can drain it without a fork. Absent variable, absent
# feature: every other system polls exactly as before.
SNNI_MARKER_FD=""
if [ -n "${SNNI_MARKER_FILE:-}" ] && [ -r "${SNNI_MARKER_FILE}" ]; then
    exec {SNNI_MARKER_FD}< "$SNNI_MARKER_FILE"
    echo "poller: following markers in $SNNI_MARKER_FILE" >&2
fi
MARKER_PENDING=""

while kill -0 "$PID" 2>/dev/null; do
"""
if s.count(OLD3) != 1:
    sys.exit(f"FAILED: the main loop header matched {s.count(OLD3)} times, expected 1")
s = s.replace(OLD3, NEW3, 1)

# 5. Drain new marker lines, then sample once for the last one seen.
OLD4 = """    echo "$TS $RSS ${VPK:-0} ${HWM:-0}" >> "$OUTDIR/vmrss.log"
"""
NEW4 = """    echo "$TS $RSS ${VPK:-0} ${HWM:-0}" >> "$OUTDIR/vmrss.log"

    # Drain whatever the program appended since the last poll. Only `MEM|<name>|` lines are phase
    # markers; the file also carries SEEDED, WORKLOAD and LEVER lines, which are announcements.
    if [ -n "$SNNI_MARKER_FD" ]; then
        while read -r -u "$SNNI_MARKER_FD" _mline; do
            case "$_mline" in
                MEM\\|*) _m="${_mline#MEM|}"; MARKER_PENDING="${_m%%|*}" ;;
            esac
        done
    fi
    if [ -n "$MARKER_PENDING" ]; then
        snapshot_peak "$TS" "$RSS" "$MARKER_PENDING"
        MARKER_PENDING=""
    fi
"""
if s.count(OLD4) != 1:
    sys.exit(f"FAILED: the vmrss.log write matched {s.count(OLD4)} times, expected 1")
s = s.replace(OLD4, NEW4, 1)

open(P, "w", encoding="utf-8").write(s)
print(f"  patched: one decomposition per phase marker ({P})")
print("  enable  : SNNI_MARKER_FILE=<run>/markers.txt")
