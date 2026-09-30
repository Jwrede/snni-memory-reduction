#!/bin/bash
#SBATCH --job-name=partyderive
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/derive_party/%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/derive_party/%x-%j.out
#
# Host object table per named party pid of one measured replicate (analysis only, no measurement).
#
#   sbatch --export=ALL,SYS=shaft-cpu,STEP=s4_onnx_and_spill,RUN=1,PIDS="1181040 1181041" \
#          derive_party.sbatch
#
# Pipeline: the system's own impl/derive.sbatch (same image, extractors under $C, same --top).
# Outputs: $C/derive_party/<sys>/<step>/run<N>/objects_host_<pid>.jsonl, objects_resident_<pid>.jsonl
# (SHAFT-CPU: also objects_pycensus_<pid>.jsonl).
# Added 2026-09-29 for the maximum-over-parties convention (README.md, "Replicates").
set -uo pipefail
SYS="${SYS:?need SYS}"; STEP="${STEP:?need STEP}"; RUN="${RUN:?need RUN}"; PIDS="${PIDS:?need PIDS}"
C=/scratch/tmp/j_wred02/snni_campaign
BIND="--bind /scratch/tmp/j_wred02:/scratch/tmp/j_wred02"
case "$SYS" in
    shaft-cpu)     R=$C/shaft-cpu/results/$STEP/run$RUN;     SIF=$C/shaft-cpu/sif/shaft-cpu-s0_default.sif; TOP=20 ;;
    shaft-cpu-vit) R=$C/shaft-cpu/results/vit_$STEP/run$RUN; SIF=$C/shaft-cpu/sif/shaft-cpu-s0_default.sif; TOP=20 ;;
    shaft-gpu)     R=$C/shaft-gpu/results/$STEP/run$RUN;     SIF=/scratch/tmp/j_wred02/shaft_gpu/shaft-gpu-estream.sif; TOP=20 ;;
    llama)         R=$C/llama/results/$STEP/run$RUN;         SIF=$C/llama/sif/llama-online-$STEP.sif; TOP=15 ;;
    sigma-gpu)     R=/scratch/tmp/j_wred02/sigma_gpu/results/$STEP/run$RUN; SIF=""; TOP=20 ;;
    *) echo "FAILED: no pipeline for $SYS"; exit 2 ;;
esac
OUT=$C/derive_party/$SYS/$STEP/run$RUN
mkdir -p "$OUT"
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }
[ -d "$R" ] || { echo "FAILED: no run at $R"; exit 1; }

if [ "$SYS" = sigma-gpu ]; then
    # SIGMA-GPU binds the binary that actually ran over the image's stock copy (see its derive.sbatch).
    SIF=$(grep -aoE 'image=[^[:space:]]+' "$R/RUN_META" | head -1 | cut -d= -f2-)
    RUNBIN=$(grep -m1 '^sigma_binary=' "$R/RUN_META" | cut -d= -f2-)
    [ -r "$RUNBIN" ] || { echo "FAILED: RUN_META names no readable sigma_binary"; exit 3; }
    BIND="$BIND --bind $RUNBIN:/sigma/EzPC/GPU-MPC/experiments/sigma/sigma"
fi
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 1; }
echo "run=$R image=$SIF out=$OUT"
grep -h "leader=" "$R/poller_attach.log"

RC=0
for PID in $PIDS; do
    P=$R/poll_$PID
    for f in "$P/peaks/resident.txt" "$P/peaks/pm.maps" "$P/peaks/peak.smaps"; do
        [ -s "$f" ] || { echo "FAILED pid=$PID: missing or empty $f"; RC=1; continue 2; }
    done
    HWM=$(awk -F= '/^# hwm_kb=/{print $2}' "$P/vmrss.log" | tail -1)
    echo "=== pid=$PID trace=$(basename "$P") hwm_kb=${HWM:-?}"
    RES=$OUT/objects_resident_$PID.jsonl
    HEAP=$RES
    case "$SYS" in
        shaft-cpu|shaft-cpu-vit)
            apptainer exec $BIND "$SIF" python3 "$C/attrib_resident/resolve_resident.py" \
                "$P/peaks/resident.txt" "$P/peaks/pm.maps" "$RES" \
                --exec "apptainer exec $BIND $SIF" --top $TOP || { RC=1; continue; }
            CENSUS=$R/pycensus_pid$PID.txt
            if [ -s "$CENSUS" ]; then
                PEAKKB=$(awk '/polled_peak_kb=/{split($0,a,"="); print a[2]}' "$R/PEAK.txt" | head -1)
                python3 "$C/attrib_pycensus/pycensus_objects.py" "$CENSUS" "$OUT/objects_pycensus_$PID.jsonl" \
                    ${PEAKKB:+--peak-kb "$PEAKKB"} && HEAP=$OUT/objects_pycensus_$PID.jsonl
            else
                echo "WARNING pid=$PID: no census $CENSUS; the heap side stays the recorder's"
            fi
            apptainer exec $BIND "$SIF" python3 "$C/smaps_objects.py" "$P/peaks" \
                "$OUT/objects_host_$PID.jsonl" --heap "$HEAP" || { RC=1; continue; } ;;
        shaft-gpu)
            apptainer exec $BIND "$SIF" python3 "$C/attrib_resident/resolve_resident.py" \
                "$P/peaks/resident.txt" "$P/peaks/pm.maps" "$RES" \
                --exec "apptainer exec $BIND $SIF" --top $TOP || { RC=1; continue; }
            apptainer exec $BIND "$SIF" python3 "$C/smaps_objects.py" "$P/peaks" \
                "$OUT/objects_host_$PID.jsonl" --heap "$HEAP" || { RC=1; continue; } ;;
        llama)
            python3 "$C/attrib_resident/resolve_resident.py" \
                "$P/peaks/resident.txt" "$P/peaks/pm.maps" "$RES" \
                --exec "apptainer exec $SIF" --top $TOP || { RC=1; continue; }
            python3 "$C/smaps_objects.py" "$P/peaks" "$OUT/objects_host_$PID.jsonl" \
                --heap "$HEAP" || { RC=1; continue; } ;;
        sigma-gpu)
            python3 "$C/attrib_resident/resolve_resident.py" \
                "$P/peaks/resident.txt" "$P/peaks/pm.maps" "$RES" \
                --exec "apptainer exec $BIND $SIF" --top $TOP || { RC=1; continue; }
            python3 "$C/smaps_objects.py" "$P/peaks" "$OUT/objects_host_$PID.jsonl" \
                --heap "$HEAP" || { RC=1; continue; } ;;
    esac
    echo "DERIVED pid=$PID -> $OUT/objects_host_$PID.jsonl"
done
[ "$RC" = 0 ] && echo "DERIVE_OK $SYS $STEP run$RUN pids=$PIDS" || echo "DERIVE_INCOMPLETE $SYS $STEP run$RUN"
exit $RC
