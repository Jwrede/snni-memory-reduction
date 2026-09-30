#!/bin/bash
#SBATCH --job-name=sigmaderive
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
# Time limit: measured derives take 1 to 12 s (longer requests delayed scheduling, job 45853820).
#SBATCH --output=/scratch/tmp/j_wred02/sigma_gpu/derive-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/sigma_gpu/derive-%x-%j.out

# One measured run -> objects_vram.jsonl and objects_host.jsonl. Two parties on one card: tables from
# the published peak's party, passed to derive_host_objects.sh explicitly. 32G (8G OOM'd once).
set -uo pipefail
C=/scratch/tmp/j_wred02/snni_campaign
B=/scratch/tmp/j_wred02/sigma_gpu
STEP="${STEP:?need STEP}"
IMG_STEP="${STEP%_pmrec}"; IMG_STEP="${IMG_STEP%_attrib}"

# Results directory resolved here, before anything reads it: a copy-paste from other systems
# once referenced it as $R (not $CLEAN) thirty lines later, killing every derive under set -u.
CLEAN=$B/results/$STEP${RUN:+/run$RUN}
# Image comes from the run's own record, not the step name: a probe's actual image is whatever
# the run fell back to. Copying the campaign script's p_*|x_* fallback here would send a step
# with its own image to the wrong binary whenever it's briefly missing.
if [ -z "${SIF:-}" ] && [ -r "$CLEAN/RUN_META" ]; then
    SIF=$(grep -aoE 'image=[^[:space:]]+' "$CLEAN/RUN_META" 2>/dev/null | head -1 | cut -d= -f2-)
    [ -r "${SIF:-}" ] || SIF=""
fi
SIF="${SIF:-$B/sigma-gpu-$IMG_STEP.sif}"
[ -r "$SIF" ] || SIF="$B/sigma-gpu-s0_default.sif"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
# apptainer does NOT auto-bind /scratch here, and the extractors live on /scratch while having to
# run INSIDE the image, because resolving symbols needs the image's own binary and libraries.
BIND="--bind /scratch/tmp/j_wred02:/scratch/tmp/j_wred02"


# Binds the binary that actually ran, over the image's stock copy: the derive previously omitted
# this, so addr2line resolved against a different program with no debug info. Caught when a
# rebuild's object table came back byte-identical (impossible across a recompile).
RUNBIN=$(grep -m1 '^sigma_binary=' "$CLEAN/RUN_META" 2>/dev/null | cut -d= -f2-)
if [ -n "$RUNBIN" ] && [ -r "$RUNBIN" ]; then
    BIND="$BIND --bind $RUNBIN:/sigma/EzPC/GPU-MPC/experiments/sigma/sigma"
    echo "symbolising against the measured binary: $RUNBIN"
else
    echo "FAILED: RUN_META names no readable sigma_binary, so the symboliser would resolve" >&2
    echo "        against the image's stock program and every name in the table would belong" >&2
    echo "        to a different binary." >&2
    exit 3
fi
[ -d "$CLEAN" ] || { echo "ERROR: no clean run at $CLEAN" >&2; exit 2; }

# Which run carries the tables is read from RUN_META's combined_channel=1, not guessed from
# directory layout: a plain fallback once paired a combined run's peak with a three-day-old
# separate run's table, built from a different binary (7 rows published 2026-08-21).
COMBINED=$(grep -c '^combined_channel=1$' "$CLEAN/RUN_META" 2>/dev/null || true)
if [ "${COMBINED:-0}" -ge 1 ]; then
    DEV=$CLEAN
    HOST=$CLEAN
    echo "channels=combined (RUN_META says so): device and host tables both come from $CLEAN"
else
    DEV=$B/results/${STEP}_attrib${RUN:+/run$RUN}
    HOST=$B/results/${STEP}_pmrec${RUN:+/run$RUN}
    echo "channels=separate: device from $DEV, host from $HOST"
fi

# Published party is the freeze leader (party 0, glob order); poll directory name is read from
# the poller's own log line, not rebuilt (rebuilding it once killed six SHARK-dealer derives).
lead_dir() {   # <run dir> -> the leader's poll directory, or empty
    local d
    d=$(sed -n 's/.*(leader=1)[[:space:]]*->[[:space:]]*//p' "$1/poller_attach.log" 2>/dev/null | tail -1)
    [ -n "$d" ] && [ -d "$d" ] && { echo "$d"; return 0; }
    # Fall back to party 0's pid, and say so, rather than to "the first peaks/ anywhere under here".
    local p0
    p0=$(tr -dc '0-9' < "$1/party0.pid" 2>/dev/null)
    [ -n "$p0" ] && [ -d "$1/poll_$p0" ] && { echo "$1/poll_$p0"; return 0; }
    return 1
}

LEAD_CLEAN=$(lead_dir "$CLEAN") || { echo "ERROR: no leader trace under $CLEAN" >&2; exit 2; }
P0=$(tr -dc '0-9' < "$CLEAN/party0.pid" 2>/dev/null)
P1=$(tr -dc '0-9' < "$CLEAN/party1.pid" 2>/dev/null)
# freeze_mode=maxparty (tools/palma/measure.sh, 2026-09-29): the party holding the maximum over
# the parties is captured with every party stopped, so the heavier party's own capture is frozen at
# its peak and is the one decomposed. Echoes that party's poll directory; returns 1 for any other
# run, and also when the heavier party has no capture (its peak a spike the poller never saw as
# the maximum), which is said, and the default choice stands.
maxparty_dir() {   # <run dir>
    local run="$1" d best="" bk=0 kb
    grep -q '^freeze_mode=maxparty$' "$run/RUN_META" 2>/dev/null || return 1
    for d in "$run"/poll_*; do
        [ -d "$d" ] || continue
        kb=$(awk 'BEGIN{m=0}
                  /^#/{ if (match($0,/hwm_kb=[0-9]+/)) { v=substr($0,RSTART+7,RLENGTH-7)+0; if (v>m) m=v }; next }
                  { if ($4+0 > m) m=$4+0 }
                  END{print m}' "$d/vmrss.log" 2>/dev/null)
        [ "${kb:-0}" -gt "$bk" ] && { bk="$kb"; best="$d"; }
    done
    [ -n "$best" ] || return 1
    if [ ! -s "$best/peaks/resident.txt" ]; then
        echo "NOTE: freeze_mode=maxparty but the heavier party $(basename "$best") has no capture;" \
             "the default party choice stands" >&2
        return 1
    fi
    echo "freeze_mode=maxparty: decomposing the heavier party $(basename "$best") (VmHWM $bk kB)" >&2
    cat "$best/peaks/CAPTURE" >&2 2>/dev/null
    echo "$best"
}
if MP=$(maxparty_dir "$CLEAN"); then
    LEAD_CLEAN="$MP"
    echo "published process = ${MP##*/poll_} (the heavier party, captured frozen at its peak under freeze_mode=maxparty)" \
        | tee "$CLEAN/published_party.txt"
else
echo "published process = $P0 (party 0, the freeze leader and the party whose output is the gate)" \
    | tee "$CLEAN/published_party.txt"
fi

# Which party is taller is measured and recorded, not assumed: if party 1 is materially above
# party 0, published peak and decomposition would describe different processes.
{
    echo "# process VmHWM_kB"
    for P in "$CLEAN"/poll_*; do
        [ -d "$P" ] || continue
        awk 'BEGIN{m=0}
             /^#/{ if (match($0,/hwm_kb=[0-9]+/)) { v=substr($0,RSTART+7,RLENGTH-7)+0; if (v>m) m=v } ; next }
             { if ($4+0 > m) m=$4+0; if ($2+0 > m) m=$2+0 }
             END{printf "%s %d\n", d, m}' d="$(basename "$P")" "$P/vmrss.log" 2>/dev/null
    done
} | tee "$CLEAN/party_peaks.txt"

RC=0

# --- device pool ---------------------------------------------------------------------------------
# cudarec writes one table per party; the leader's (party 0) pairs with the published peak. Run
# root is also tried, since that's where single-process GPU systems put it.
DEV_LEAD=$(lead_dir "$DEV" 2>/dev/null || true)
for cand in "$DEV/dev_party0" "$DEV" "$DEV_LEAD"; do
    [ -n "$cand" ] || continue
    if [ -s "$cand/device_table.txt" ] && [ -s "$cand/device.maps" ]; then
        # Resolver runs on host, symboliser in the image (neither GPU image carries python3; cost
        # MOAI-GPU nine runs before found). --inline avoids collapsing inlined allocations onto
        # the wrapper function they were inlined from (this build's largest entry was the macro).
        python3 "$C/attrib_device/resolve_device.py" \
            "$cand/device_table.txt" "$cand/device.maps" "$CLEAN/objects_vram.jsonl" \
            --exec "apptainer exec $BIND $SIF" --inline \
            && { echo "wrote objects_vram.jsonl from $cand"; break; } \
            || { echo "ERROR: device resolve failed on $cand" >&2; RC=3; break; }
    fi
done
if [ ! -s "$CLEAN/objects_vram.jsonl" ] && [ "$RC" = 0 ]; then
    echo "ERROR: no device table under $DEV (nor under the leader's trace). This system's device" >&2
    echo "       channel is cudarec, and an empty table is what a wrong interposition set looks" >&2
    echo "       like, not a quiet system." >&2
    RC=3
fi

# --- host pool -----------------------------------------------------------------------------------
LEAD_HOST=$(lead_dir "$HOST" 2>/dev/null || true)
MPH=$(maxparty_dir "$HOST" 2>/dev/null) && LEAD_HOST="$MPH"
if [ -n "$LEAD_HOST" ]; then
    # Fifth argument ($BIND, the measured binary) was previously omitted for the host path, so
    # addr2line resolved against the stock binary instead of what ran (misnamed a 32 GiB object
    # as SigmaPeer::wait() rather than cpuMalloc).
    bash "$C/derive_host_objects.sh" "$LEAD_CLEAN" "$LEAD_HOST" "$SIF" "$CLEAN/objects_host.jsonl" \
        "$BIND" || { echo "ERROR: host derive failed" >&2; RC=4; }
elif [ -d "$HOST" ]; then
    echo "ERROR: a pmrec run exists at $HOST but its leader trace could not be identified, and" >&2
    echo "       taking whichever peaks/ turns up first would pair party 1's decomposition with" >&2
    echo "       party 0's peak." >&2
    RC=4
else
    echo "ERROR: no pmrec run at $HOST" >&2
    RC=4
fi

# --- what came out -------------------------------------------------------------------------------
for f in objects_vram.jsonl objects_host.jsonl; do
    if [ -s "$CLEAN/$f" ]; then
        echo "--- $f"
        python3 - "$CLEAN/$f" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1]).read().splitlines()[0])
objs = sorted(d["objects"].items(), key=lambda kv: -kv[1])
print(f"  source={d['source']}  total={d['rss_kb']*1024/1e9:.2f} GB  objects={len(objs)}")
for k, v in objs[:8]:
    print(f"    {v/1e6:9.1f} MB  {k[:80]}")
PY
    else
        echo "--- $f MISSING"
    fi
done
exit $RC
