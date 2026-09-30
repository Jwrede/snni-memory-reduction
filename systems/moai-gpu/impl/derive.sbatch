#!/bin/bash
#SBATCH --job-name=mgderive
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/derive-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/derive-%x-%j.out
#
# One replicate's three channels -> two canonical object tables (on the cluster).
#
#   sbatch --export=ALL,STEP=s0_default,RUN=1 derive.sbatch
#
#   objects_vram.jsonl  device: cudarec's peak table, symbolised against the run's own maps
#   objects_host.jsonl  host: libraries from the CLEAN run's smaps + objects in RESIDENT bytes
#                       from the pmrec run, with whatever neither names as anon:residual
#
# 32G (8G OOM-killed on a smaller sif).
set -uo pipefail
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/moai-gpu
STEP="${STEP:?need STEP}"
SIF="${SIF:-$B/sif/moai-gpu-$STEP.sif}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
# apptainer does not auto-bind /scratch here, and the extractors must run inside the image (they
# need its own binary/libraries); without the bind, only the python half reports "No such file".
BIND="--bind /scratch/tmp/j_wred02:/scratch/tmp/j_wred02"

CLEAN=$B/results/$STEP${RUN:+/run$RUN}
# Both arrangements are read, neither assumed (2026-08-21): reads RUN_META's own record of whether
# this run was combined-channel or three-run (clean/_attrib/_pmrec), rather than falling back to
# stale _attrib/_pmrec directories left over from the three-run era (SIGMA-GPU, 2026-08-22).
COMBINED=$(grep -c '^combined_channel=1$' "$CLEAN/RUN_META" 2>/dev/null || true)
if [ "${COMBINED:-0}" -ge 1 ]; then
    DEV=$CLEAN
    HOST=$CLEAN
    echo "channels=combined (RUN_META says so): device and host tables both come from $CLEAN"
else
    DEV=$B/results/${STEP}_attrib${RUN:+/run$RUN}
    [ -d "$DEV" ] || DEV=$CLEAN
    HOST=$B/results/${STEP}_pmrec${RUN:+/run$RUN}
    [ -d "$HOST" ] || HOST=$CLEAN
    echo "channels=separate: device from $DEV, host from $HOST"
fi
[ -d "$CLEAN" ] || { echo "ERROR: no clean run at $CLEAN" >&2; exit 2; }

RC=0

# --- device pool -------------------------------------------------------------------------------
# Single process; still must pair the table with its own maps snapshot (addresses are run-relative).
if [ -s "$DEV/device_table.txt" ] && [ -s "$DEV/device.maps" ]; then
    # Resolver runs on the host, symboliser (via --exec) in the image: this image has no python3.
    # --inline names a site by the function the allocation was inlined into, not the wrapper it
    # came from; without it, the largest entry is the generic `check<cudaError>` wrapper (49.3%).
    python3 "$C/attrib_device/resolve_device.py" \
        "$DEV/device_table.txt" "$DEV/device.maps" "$CLEAN/objects_vram.jsonl" \
        --exec "apptainer exec $BIND $SIF" --inline \
        && echo "wrote objects_vram.jsonl" \
        || { echo "ERROR: device resolve failed" >&2; RC=3; }
else
    echo "ERROR: no device table under $DEV. This system's device channel is cudarec, and an" >&2
    echo "       empty table is what a wrong interposition set looks like, not a quiet system." >&2
    RC=3
fi

# --- host pool ---------------------------------------------------------------------------------
if [ -d "$HOST" ]; then
    bash "$C/derive_host_objects.sh" "$CLEAN" "$HOST" "$SIF" "$CLEAN/objects_host.jsonl" \
        || { echo "ERROR: host derive failed" >&2; RC=4; }
else
    echo "ERROR: no pmrec run at $HOST" >&2
    RC=4
fi

# --- what came out -----------------------------------------------------------------------------
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
