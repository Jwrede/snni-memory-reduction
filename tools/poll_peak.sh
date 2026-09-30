#!/bin/bash
# Peak-triggered memory poller for CPU systems: running RSS maximum, snapshot at each new maximum.
#
# Usage: poll_peak.sh <PID> [interval_ms] [outdir]
#
# Output in <outdir>:
#   vmrss.log                      ts rss vmpeak vmhwm, every sample
#   rollup.log                     ts rss anon file shmem
#   peaks/peak.smaps, peak.status, PEAK   highest maximum, rewritten at each new maximum
#   peaks/<ts>.smaps, INDEX        kept on a rise >= PEAK_TRIGGER_PCT (default 2%)
#   peaks/resident.txt, PM_INDEX, pm.maps  per-call-site resident bytes (if SNNI_PM_SAMPLE is set)
set -uo pipefail

PID="${1:?Usage: poll_peak.sh <PID> [interval_ms] [outdir]}"
INTERVAL_MS="${2:-100}"
OUTDIR="${3:-.}"
TRIGGER_PCT="${PEAK_TRIGGER_PCT:-2}"

mkdir -p "$OUTDIR/peaks"
SLEEP_SEC=$(awk "BEGIN {printf \"%.3f\", ${INTERVAL_MS}/1000}")

echo "# timestamp_ms VmRSS_kB VmPeak_kB VmHWM_kB" > "$OUTDIR/vmrss.log"
echo "# timestamp_ms Rss_kB Anonymous_kB Filemapped_kB Shmem_kB" > "$OUTDIR/rollup.log"
echo "# timestamp_ms rss_kb" > "$OUTDIR/peaks/INDEX"

MAX=0
MAXHWM=0
LAST_TRIGGER=0
LAST_CENSUS=0
LAST_TS=0
# Resident-sampler throttle as a DUTY CYCLE (SNNI_PM_DUTY_PCT of wall), self-tuning because a
# sample's cost scales with the address space, which grows during the run. Duty bounds COST only; a
# second gate bounds ACCURACY: a rise > PM_MAX_GAP_PROMILLE above the last sample forces a sample.
# %d in the table path expands to the polled pid (matching pmrec.so), so multi-party processes do
# not truncate one shared table.
if [ -n "${SNNI_PM_TABLE:-}" ]; then
    SNNI_PM_TABLE="${SNNI_PM_TABLE//%d/$PID}"
    export SNNI_PM_TABLE
fi
# The SEAL pool recorder writes the same table format into its own mapping (MOAI-CPU only; empty
# elsewhere, so the lines below are skipped).
if [ -n "${SNNI_POOLREC_TABLE:-}" ]; then
    SNNI_POOLREC_TABLE="${SNNI_POOLREC_TABLE//%d/$PID}"
    export SNNI_POOLREC_TABLE
fi
PM_DUTY_PCT="${SNNI_PM_DUTY_PCT:-5}"
PM_MAX_GAP_PROMILLE="${SNNI_PM_MAX_GAP_PROMILLE:-10}"
PM_NEXT_TS=0
PM_LAST_RSS=0
# Processes to stop, default the polled one. Multi-party must list EVERY party: stopping one peer
# while others run is a protocol fault, not a pause. One kill stops them within microseconds.
FREEZE_PIDS="${SNNI_PM_FREEZE_PIDS:-$PID}"
TOTAL_PAUSE_MS=0
# If the poller dies between SIGSTOP and SIGCONT the processes stay stopped forever (burning walltime
# until SLURM kills them). The trap sends SIGCONT on any exit path (harmless if already running).
if [ -n "${SNNI_PM_FREEZE:-}" ]; then
    # shellcheck disable=SC2086
    trap 'kill -CONT $FREEZE_PIDS 2>/dev/null' EXIT INT TERM
fi
# Max-over-parties trigger (SNNI_PM_MAXPARTY_ROLE, set only by snni_start_poller in measure.sh under
# the campaign variable SNNI_PM_FREEZE_MODE=maxparty, 2026-09-29). The published peak of a
# multi-party run is the maximum over the parties, so the leader reads every party's VmRSS at each
# poll and, when the maximum over the parties reaches a new high, runs the unchanged snapshot_peak
# for the party that holds it (every party stopped by the one kill, as before), writing into that
# party's own poll directory (SNNI_PM_MAXPARTY_OUT, `%d` = pid). The heavier party's last capture
# is then frozen at its own peak; while the leader holds the maximum, the leader's captures are
# exactly the leader-only ones. Followers keep their traces and take no captures at new maxima.
# Unset, nothing below changes.
MP_ROLE="${SNNI_PM_MAXPARTY_ROLE:-}"
MP_MAX=0
declare -A MP_RSS=() MP_LAST_RSS=()
case "$MP_ROLE" in
    "" | follower) ;;
    leader)
        : "${SNNI_PM_FREEZE:?maxparty leader needs SNNI_PM_FREEZE}"
        : "${SNNI_PM_MAXPARTY_OUT:?maxparty leader needs SNNI_PM_MAXPARTY_OUT (with %d)}"
        for _p in $FREEZE_PIDS; do mkdir -p "${SNNI_PM_MAXPARTY_OUT//%d/$_p}/peaks"; done
        echo "# ts_ms max_pid max_rss_kb pause_ms pid:rss_kb ..." > "$OUTDIR/peaks/MAXPARTY_INDEX"
        ;;
    *) echo "poller: unknown SNNI_PM_MAXPARTY_ROLE=$MP_ROLE" >&2; exit 2 ;;
esac
# SNNI_PM_FULLSCAN=1: pmsample scans every readable mapping, so the unnamed remainder is LOCATED
# rather than the subtraction anon-minus-named-heap. Costs address space, not RSS (+37% on 600 MB
# but 36x on a 512 GiB PROT_NONE reservation, the torch/CUDA shape), so unreadable mappings are
# skipped and the skipped total printed; too long a wait is refused by make_steps.py as MISALIGNED.
[ -n "${SNNI_PM_FULLSCAN:-}" ] && export SNNI_PM_FULLSCAN

snapshot() {
    local ts="$1" rss="$2"
    cp "/proc/$PID/smaps"  "$OUTDIR/peaks/${ts}.smaps"  2>/dev/null
    cp "/proc/$PID/status" "$OUTDIR/peaks/${ts}.status" 2>/dev/null
    echo "$ts $rss" >> "$OUTDIR/peaks/INDEX"
    return 0
}

# Written via a temp file and renamed, so a process that dies mid-copy leaves the previous
# complete snapshot rather than a truncated one that would parse as a smaller peak.
snapshot_peak() {
    local ts="$1" rss="$2" marker="${3:-}"
    # Trace-only mode: log the timeline, take no snapshots (which are slowest on a rising ramp), so
    # the loop runs at a few ms to measure how long a peak persists.
    if [ -n "${SNNI_PM_TRACE_ONLY:-}" ]; then
        echo "$ts $rss" > "$OUTDIR/peaks/PEAK"
        return 0
    fi
    # One counter per marker sample, bumped here (not where files are written) so it stays in step
    # with MARKER_INDEX; printf -v is a builtin, no fork.
    if [ -n "$marker" ]; then printf -v MARKER_SEQ '%02d' "$(( 10#$MARKER_SEQ + 1 ))"; fi
    # rss_pre/rss_post bracket the snapshot sequence, so a moving peak (a 21 ms spike vs a slow ramp)
    # is detectable: the three instruments together take tens of ms and would otherwise each describe
    # a different program (d7: object table 193 MB while the scan beside it found 31 MB resident total).
    local rss_pre rss_post froze=0 fz0 fz1 pause_ms=0

    # Freeze-then-measure (SNNI_PM_FREEZE=1): every instrument here is a sample that races the
    # program (attribution came out 104%, 535%, 1808% when the table was at the peak and the snapshot
    # after a 96% collapse). Polling faster only moves the threshold; stopping the process removes the
    # race, so objects/mappings/libraries describe one instant. It perturbs wall time only, and that
    # is logged and totalled so it can be subtracted from the runtime axis.
    if [ -n "${SNNI_PM_FREEZE:-}" ]; then
        fz0=$(date +%s%3N)
        # shellcheck disable=SC2086
        kill -STOP $FREEZE_PIDS 2>/dev/null && froze=1
        # SIGSTOP is asynchronous: kill() returns when the signal is queued, so /proc read straight
        # after can catch a running process (SHAFT-GPU: RSS fell 2670008 -> 2370180 kB inside a
        # supposedly frozen window). Wait for state 'T' in /proc/<pid>/stat field 3.
        if [ "$froze" -eq 1 ]; then
            local w p st
            for w in $(seq 1 200); do
                local pending=0
                for p in $FREEZE_PIDS; do
                    st=$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null)
                    [ -z "$st" ] && continue          # gone: nothing to wait for
                    case "$st" in T|t) ;; *) pending=1 ;; esac
                done
                [ "$pending" -eq 0 ] && break
                sleep 0.001
            done
        fi
    fi

    rss_pre=$(awk '/^VmRSS:/{print $2; exit}' "/proc/$PID/status" 2>/dev/null)

    # Order: the resident table goes first (it must be at the peak; residency changes by the ms),
    # then smaps (mostly stable file-backed libraries). A marker sample ignores the duty gate: it
    # asks for the composition at a declared instant, so skipping it would name the marker over an
    # older sample.
    if [ -n "${SNNI_PM_TABLE:-}" ] && [ -x "${SNNI_PM_SAMPLE:-}" ] && {
           [ -n "$marker" ] ||
           [ "$ts" -ge "$PM_NEXT_TS" ] || [ "$PM_LAST_RSS" -eq 0 ] ||
           [ "$(( (rss - PM_LAST_RSS) * 1000 / PM_LAST_RSS ))" -ge "$PM_MAX_GAP_PROMILLE" ]; }
    then
        # Separate destinations: resident.txt/resident_pool.txt are the PEAK decomposition
        # derive.sbatch reads, so a marker sample must not write them (p_prm2 died on "no rows in the
        # resident table" when a marker left after_bootstrap4 there).
        local pt0 pt1 pcost dst dstpool
        if [ -n "$marker" ]; then
            # Sequence number is part of the name: both layers emit the same marker names, so a
            # name-keyed file let layer 1 overwrite layer 0 (p_prm2). The number is the MARKER_INDEX line.
            dst="$OUTDIR/peaks/marker_${MARKER_SEQ}_${marker}.txt"
            dstpool="$OUTDIR/peaks/marker_${MARKER_SEQ}_${marker}_pool.txt"
        else
            dst="$OUTDIR/peaks/resident.txt"
            dstpool="$OUTDIR/peaks/resident_pool.txt"
        fi
        pt0=$(date +%s%3N)
        if "$SNNI_PM_SAMPLE" "$PID" "$SNNI_PM_TABLE" "$OUTDIR/peaks/pm.maps" \
                > "$dst.tmp" 2>/dev/null; then
            sed -i "1i # sample_ts_ms=$ts sample_rss_kb=$rss" "$dst.tmp"
            mv "$dst.tmp" "$dst"
            [ -z "$marker" ] && PM_LAST_RSS="$rss"
        fi
        # The pool table, in the same frozen window and gate. On MOAI-CPU frees go back to SEAL's
        # pool, so the host table names who grew the pool; the recording pool names what is alive.
        if [ -n "${SNNI_POOLREC_TABLE:-}" ] && [ -s "$SNNI_POOLREC_TABLE" ]; then
            if "$SNNI_PM_SAMPLE" "$PID" "$SNNI_POOLREC_TABLE" "$OUTDIR/peaks/pool.maps" \
                    > "$dstpool.tmp" 2>/dev/null; then
                sed -i "1i # sample_ts_ms=$ts sample_rss_kb=$rss source=seal_pool_recorder" \
                    "$dstpool.tmp"
                mv "$dstpool.tmp" "$dstpool"
                # Keep this sample, not only the last: ln is one directory entry (frozen window not
                # measurably longer) and the next sample arrives as a fresh inode via mv.
                ln -f "$dstpool" "$OUTDIR/peaks/pool_$ts.txt" 2>/dev/null || true
            fi
        fi
        pt1=$(date +%s%3N)
        pcost=$(( pt1 - pt0 ))
        PM_NEXT_TS=$(( pt1 + pcost * (100 - PM_DUTY_PCT) / (PM_DUTY_PCT > 0 ? PM_DUTY_PCT : 100) ))
        echo "$ts $rss $pcost" >> "$OUTDIR/peaks/PM_INDEX"
    fi

    # A marker sample is not a peak: the peak record is written only when RSS reached a new maximum.
    if [ -z "$marker" ]; then
        cp "/proc/$PID/smaps"  "$OUTDIR/peaks/.peak.smaps.tmp"  2>/dev/null &&
            mv "$OUTDIR/peaks/.peak.smaps.tmp"  "$OUTDIR/peaks/peak.smaps"
        cp "/proc/$PID/status" "$OUTDIR/peaks/.peak.status.tmp" 2>/dev/null &&
            mv "$OUTDIR/peaks/.peak.status.tmp" "$OUTDIR/peaks/peak.status"
        echo "$ts $rss" > "$OUTDIR/peaks/PEAK"
    else
        echo "$ts $rss $marker" >> "$OUTDIR/peaks/MARKER_INDEX"
    fi

    rss_post=$(awk '/^VmRSS:/{print $2; exit}' "/proc/$PID/status" 2>/dev/null)

    if [ "$froze" -eq 1 ]; then
        # shellcheck disable=SC2086
        kill -CONT $FREEZE_PIDS 2>/dev/null
        fz1=$(date +%s%3N)
        pause_ms=$(( fz1 - fz0 ))
        TOTAL_PAUSE_MS=$(( TOTAL_PAUSE_MS + pause_ms ))
        echo "$ts $pause_ms" >> "$OUTDIR/peaks/PAUSE_INDEX"
    fi

    # ts, the labelling RSS, the pre/post readings, and the hold time. With freeze on, rss_pre and
    # rss_post must be identical (the assertion the design rests on), so it is recorded.
    echo "$ts $rss ${rss_pre:-0} ${rss_post:-0} ${pause_ms}" > "$OUTDIR/peaks/SNAPSHOT_SPAN"
    # The recorder's table is a file-backed mapping in the measured process, so it is in RSS and is
    # subtracted by IDENTIFYING the mapping (exact, survives a table-size change). Matched by PREFIX
    # and summed over ALL recorder tables (pm/dev/pool), so a device table's devtable_ is subtracted
    # too (25,804 kB on MOAI-GPU, 1.2% of the peak, was missed by a single-variable match).
    if [ -f "$OUTDIR/peaks/peak.smaps" ]; then
        awk '
            /^[0-9a-f]+-[0-9a-f]+ / { inmap = ($0 ~ /\/(pm|dev|pool)table_/) ; next }
            inmap && /^Rss:/ { tot += $2; inmap = 0 }
            END { print tot + 0 }
        ' "$OUTDIR/peaks/peak.smaps" > "$OUTDIR/peaks/INSTRUMENT_KB" 2>/dev/null
    fi
    return 0
}

# VmRSS of one pid into MP_R, without a fork (0 if the process is gone).
mp_vmrss() {
    local k v
    MP_R=0
    [ -r "/proc/$1/status" ] || return 0
    while read -r k v _; do
        [ "$k" = "VmRSS:" ] && { MP_R="$v"; return 0; }
    done < "/proc/$1/status" 2>/dev/null
    return 0
}

# The maxparty leader's capture of party <top>: snapshot_peak with the per-party state swapped in
# (pid, output directory, recorder table, the RSS of that party's last resident sample). The duty
# cycle PM_NEXT_TS stays shared, since it bounds the sampler's total cost. The pool recorder is
# single-party (MOAI-CPU), so it is sampled only when the leader itself holds the maximum.
maxparty_capture() {
    local ts="$1" top="$2" p rec="" pz="$TOTAL_PAUSE_MS"
    local s_pid="$PID" s_out="$OUTDIR" s_tbl="${SNNI_PM_TABLE:-}" s_pool="${SNNI_POOLREC_TABLE:-}"
    local s_last="$PM_LAST_RSS"
    PID="$top"
    OUTDIR="${SNNI_PM_MAXPARTY_OUT//%d/$top}"
    [ -n "$s_tbl" ] && SNNI_PM_TABLE="${SNNI_PM_MAXPARTY_TABLE//%d/$top}"
    [ "$top" = "$s_pid" ] || SNNI_POOLREC_TABLE=""
    PM_LAST_RSS="${MP_LAST_RSS[$top]:-0}"
    snapshot_peak "$ts" "${MP_RSS[$top]}"
    MP_LAST_RSS[$top]="$PM_LAST_RSS"
    echo "capture=maxparty frozen_by=$s_pid ts_ms=$ts max_pid=$top max_rss_kb=${MP_RSS[$top]}" \
        > "$OUTDIR/peaks/CAPTURE"
    PID="$s_pid"; OUTDIR="$s_out"; SNNI_PM_TABLE="$s_tbl"; SNNI_POOLREC_TABLE="$s_pool"
    PM_LAST_RSS="$s_last"
    for p in $FREEZE_PIDS; do rec="$rec $p:${MP_RSS[$p]}"; done
    echo "$ts $top ${MP_RSS[$top]} $(( TOTAL_PAUSE_MS - pz ))$rec" >> "$OUTDIR/peaks/MAXPARTY_INDEX"
    return 0
}

# The marker file, held open so the loop can drain it without a fork. Absent variable, absent feature.
SNNI_MARKER_FD=""
if [ -n "${SNNI_MARKER_FILE:-}" ] && [ -r "${SNNI_MARKER_FILE}" ]; then
    exec {SNNI_MARKER_FD}< "$SNNI_MARKER_FILE"
    echo "poller: following markers in $SNNI_MARKER_FILE" >&2
fi
MARKER_PENDING=""
# Counts marker samples, so a file name can carry the ORDER as well as the name.
MARKER_SEQ="00"

while kill -0 "$PID" 2>/dev/null; do
    # No fork in the polling loop: date + awk-over-/proc is two fork+exec per sample (measured medians
    # 36-63 ms against POLL_MS=15, so resolution was 2.4-4.2x coarser than campaign.env). EPOCHREALTIME
    # and a read loop over /proc do it in-process.
    TS="${EPOCHREALTIME/./}"; TS="${TS:0:${#TS}-3}"
    # VmHWM (kernel high-water) catches short spikes a poller misses (LLAMA d2: 1465928 vs 900808 kB,
    # same binary), so it decides the peak VALUE; VmRSS still gives the timeline. It is a slight lower
    # bound, not exact: the PALMA kernel sets the stored high-water from an approximate per-CPU
    # counter, so it can fall back after a spike (measured up to ~43 MB, typically under 1%).
    RSS=""; VPK=0; HWM=0
    while read -r _k _v _; do
        case "$_k" in
            VmRSS:)  RSS="$_v" ;;
            VmPeak:) VPK="$_v" ;;
            VmHWM:)  HWM="$_v" ;;
            VmSwap:) break ;;
        esac
    done < "/proc/$PID/status" 2>/dev/null
    [ -z "${RSS:-}" ] && { sleep "$SLEEP_SEC"; continue; }
    echo "$TS $RSS ${VPK:-0} ${HWM:-0}" >> "$OUTDIR/vmrss.log"

    # Drain whatever the program appended since the last poll. Only `MEM|<name>|` lines are phase
    # markers; the file also carries SEEDED, WORKLOAD and LEVER lines, which are announcements.
    if [ -n "$SNNI_MARKER_FD" ]; then
        while read -r -u "$SNNI_MARKER_FD" _mline; do
            case "$_mline" in
                MEM\|*) _m="${_mline#MEM|}"; MARKER_PENDING="${_m%%|*}" ;;
            esac
        done
    fi
    if [ -n "$MARKER_PENDING" ]; then
        snapshot_peak "$TS" "$RSS" "$MARKER_PENDING"
        MARKER_PENDING=""
    fi
    [ "${HWM:-0}" -gt "$MAXHWM" ] && MAXHWM="$HWM"

    # The maxparty leader's trigger: the maximum over the parties' VmRSS at this poll (ties to the
    # first-listed party, the leader), captured when it reaches a new high. Not once a party has
    # exited: the protocol is over, and a kill -STOP naming a vanished pid fails, so snapshot_peak
    # would not send the SIGCONT.
    if [ "$MP_ROLE" = leader ]; then
        _top="$PID"; _gone=0; MP_RSS[$PID]="$RSS"
        for _p in $FREEZE_PIDS; do
            [ "$_p" = "$PID" ] && continue
            mp_vmrss "$_p"; MP_RSS[$_p]="$MP_R"
            [ "$MP_R" -eq 0 ] && _gone=1
            [ "$MP_R" -gt "${MP_RSS[$_top]}" ] && _top="$_p"
        done
        if [ "$_gone" -eq 0 ] && [ "${MP_RSS[$_top]}" -gt "$MP_MAX" ]; then
            MP_MAX="${MP_RSS[$_top]}"
            maxparty_capture "$TS" "$_top"
        fi
    fi

    # Nothing expensive between the RSS reading and the freeze: the rollup awk used to sit here and
    # let a falling peak freeze low (SHAFT-GPU smaps 12.6% below vs the resident table 4.6%). The
    # rollup is a timeline, so it is read AFTER the capture decision.
    if [ "$RSS" -gt "$MAX" ]; then
        MAX="$RSS"
        LAST_TS="$TS"
        # Always keep the snapshot at the highest maximum: one file, overwritten. Under maxparty
        # the captures come from maxparty_capture alone.
        [ -z "$MP_ROLE" ] && snapshot_peak "$TS" "$RSS"
        # Additionally keep a timeline entry, but only on a materially higher peak, else
        # warm-up leaves hundreds of files.
        if [ "$LAST_TRIGGER" -eq 0 ] || \
           [ "$(( (RSS - LAST_TRIGGER) * 100 / (LAST_TRIGGER > 0 ? LAST_TRIGGER : 1) ))" -ge "$TRIGGER_PCT" ]; then
            snapshot "$TS" "$RSS"
            LAST_TRIGGER="$RSS"
            # When a system has an in-process census, signal it here: every new maximum passes
            # through this branch, so the last one is the peak (a wall-clock census sat at 79% of the
            # peak). Throttled by PEAK_TRIGGER_PCT, opt-in via SNNI_CENSUS_SIGNAL, sent before the freeze.
            if [ -n "${SNNI_CENSUS_SIGNAL:-}" ]; then
                kill -"$SNNI_CENSUS_SIGNAL" "$PID" 2>/dev/null \
                    && echo "$TS census_signal rss=$RSS" >> "$OUTDIR/census_signal.log"
            fi
        fi
        # A census that is too costly to repeat on every material new maximum (a Go heap dump is a
        # stop-the-world write of the whole heap) runs as a HOOK instead: opt-in via
        # SNNI_CENSUS_HOOK, only above SNNI_CENSUS_MIN_KB, throttled by its own SNNI_CENSUS_PCT.
        # The hook freezes and captures itself; its whole duration is pause, so it is added to
        # frozen_ms and the wall is corrected for it exactly as for the peak freezes above.
        # SNNI_CENSUS_PERMILLE (opt-in) sets the step in tenths of a percent and overrides
        # SNNI_CENSUS_PCT; unset, the step is SNNI_CENSUS_PCT whole percent as before.
        if [ -n "${SNNI_CENSUS_HOOK:-}" ] && [ "$RSS" -ge "${SNNI_CENSUS_MIN_KB:-0}" ] && {
               [ "$LAST_CENSUS" -eq 0 ] ||
               if [ -n "${SNNI_CENSUS_PERMILLE:-}" ]; then
                   [ "$(( (RSS - LAST_CENSUS) * 1000 / LAST_CENSUS ))" -ge "$SNNI_CENSUS_PERMILLE" ]
               else
                   [ "$(( (RSS - LAST_CENSUS) * 100 / LAST_CENSUS ))" -ge "${SNNI_CENSUS_PCT:-$TRIGGER_PCT}" ]
               fi; }
        then
            _h0=$(date +%s%3N)
            bash "$SNNI_CENSUS_HOOK" "$PID" "$TS" "$RSS" >> "$OUTDIR/census.log" 2>&1
            _h1=$(date +%s%3N)
            TOTAL_PAUSE_MS=$(( TOTAL_PAUSE_MS + _h1 - _h0 ))
            echo "$TS $(( _h1 - _h0 )) census" >> "$OUTDIR/peaks/PAUSE_INDEX"
            LAST_CENSUS="$RSS"
        fi
    fi
    # The timeline, read after any capture decision so it never delays a freeze.
    if [ -r "/proc/$PID/smaps_rollup" ]; then
        awk -v ts="$TS" '
            /^Rss:/{r=$2} /^Anonymous:/{a=$2} /^Shmem:/{s=$2}
            END{printf "%s %d %d %d %d\n", ts, r, a, r-a-s, s}
        ' "/proc/$PID/smaps_rollup" >> "$OUTDIR/rollup.log" 2>/dev/null
    fi

    sleep "$SLEEP_SEC"
done

# The true maximum is in peaks/PEAK; note in the timeline index when the thresholded channel stopped
# short of it, so the two are not confused.
if [ "$MAX" -gt 0 ] && [ "$MAX" -ne "$LAST_TRIGGER" ]; then
    echo "$LAST_TS $MAX (true peak, below trigger threshold; snapshot is in peaks/peak.smaps)" \
        >> "$OUTDIR/peaks/INDEX"
fi

# peak_kb (polled max) and hwm_kb (kernel high-water) side by side, not reconciled: a gap says the
# peak was a spike the poller missed, so anything taken at the peak describes a lower moment.
echo "# peak_kb=$MAX" >> "$OUTDIR/vmrss.log"
echo "# hwm_kb=$MAXHWM" >> "$OUTDIR/vmrss.log"
# Total hold time: the instrument's entire, exact runtime cost, subtracted from wall before a step
# is typed free/paid (a known cost is removable, like the recorder's table from the peak).
echo "# frozen_ms=$TOTAL_PAUSE_MS" >> "$OUTDIR/vmrss.log"
if [ "$MAXHWM" -gt 0 ] && [ "$MAX" -gt 0 ]; then
    echo "# polled_vs_hwm_pct=$(( (MAXHWM - MAX) * 1000 / MAXHWM ))promille" >> "$OUTDIR/vmrss.log"
fi
