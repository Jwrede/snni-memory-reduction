#!/bin/bash
#SBATCH --job-name=bbnetp
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/bb_bwlat_proxy/logs/%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/bb_bwlat_proxy/logs/%x-%j.out
#
# One cell of the BumbleBee bandwidth x latency grid, shaped in userspace (unprivileged network
# namespaces disabled on PALMA: max_net_namespaces = 0, 2026-09-30).
# Parties listen on 9530/9531; connshim.so (preload) redirects connect() to 19530/19531; shaper.py
# forwards 19530->9530, 19531->9531 with one-way delay RTT/2 and one shared token bucket at RATE.
# Measured binary: unmodified image.
#
#   sbatch --job-name=bbnetp_<cell> --export=ALL,RATE=<mbit>,RTT=<ms>,CONC=<16|1>,CELL=<name>,BB=<sif>,OUTROOT=<dir> bb_cell_proxy.sbatch
set -uo pipefail
WD=/scratch/tmp/j_wred02/bb_bwlat_proxy
RUNBB=$WD/run_bumblebee.sh
NLAYERS="${NLAYERS:-12}"
: "${RATE:?}"; : "${RTT:?}"; : "${CONC:?}"; : "${CELL:?}"; : "${BB:?}"; : "${OUTROOT:?}"
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
for f in "$BB" "$RUNBB" "$WD/shaper.py" "$WD/connshim.so" "$WD/rttprobe_proxy.py"; do
    [ -r "$f" ] || { echo "MISSING $f"; exit 2; }
done
RES=$OUTROOT/$CELL
rm -rf "$RES"; mkdir -p "$RES"
DELAY=$(python3 -c "print($RTT/2.0)")
echo "cell=$CELL rate=${RATE}mbit rtt=${RTT}ms oneway=${DELAY}ms conc=$CONC nlayers=$NLAYERS host=$(hostname) image=$BB" | tee "$RES/CELL_META.txt"
sha256sum "$BB" | cut -c1-16 >> "$RES/CELL_META.txt"

python3 -u "$WD/shaper.py" --rate-mbit "$RATE" --delay-ms "$DELAY" \
    --map 19530:9530 --map 19531:9531 --map 29999:19999 --stats "$RES/shaper_stats.json" \
    > "$RES/shaper.log" 2>&1 &
SH=$!
for _ in $(seq 1 100); do grep -q SHAPER_READY "$RES/shaper.log" 2>/dev/null && break; sleep 0.1; done
python3 "$WD/rttprobe_proxy.py" > "$RES/rtt.txt" 2>&1 || echo RTT_PROBE_FAIL > "$RES/rtt.txt"

T0=$(date +%s)
apptainer exec --bind "$RES":"$RES" --bind "$WD":"$WD" "$BB" bash -c \
    "export BB_RESULTS='$RES' BB_N_LAYERS='$NLAYERS' BB_SEQ_LEN=128 BB_MAX_CONCURRENCY='$CONC' OMP_NUM_THREADS=16 SNNI_CONNMAP='9530:19530,9531:19531' LD_PRELOAD='$WD/connshim.so'; exec bash '$RUNBB'" \
    > "$RES/outer.log" 2>&1 &
APP=$!
for _ in $(seq 1 6000); do
    [ -s "$RES/party0.pid" ] && [ -s "$RES/party1.pid" ] && break
    kill -0 "$APP" 2>/dev/null || break
    sleep 0.1
done
P0=$(cat "$RES/party0.pid" 2>/dev/null || echo); P1=$(cat "$RES/party1.pid" 2>/dev/null || echo)
echo "party0=$P0 party1=$P1" | tee -a "$RES/CELL_META.txt"
hwm() { awk '/^VmHWM:/{print $2}' "/proc/$1/status" 2>/dev/null; }
MAX0=0; MAX1=0
if [ -n "$P0" ] && [ -n "$P1" ]; then
    while kill -0 "$APP" 2>/dev/null; do
        h0=$(hwm "$P0"); h1=$(hwm "$P1")
        [ -n "${h0:-}" ] && [ "$h0" -gt "$MAX0" ] 2>/dev/null && MAX0=$h0
        [ -n "${h1:-}" ] && [ "$h1" -gt "$MAX1" ] 2>/dev/null && MAX1=$h1
        sleep 0.1
    done
else
    echo "WARNING: missing party pids; no peak" | tee -a "$RES/CELL_META.txt"
fi
wait "$APP"; RC=$?
T1=$(date +%s)
kill -TERM $SH 2>/dev/null; wait $SH 2>/dev/null
BB_WALL=$(grep -oE 'wall=[0-9]+s' "$RES/outer.log" 2>/dev/null | tail -1 | tr -dc '0-9'); [ -n "$BB_WALL" ] || BB_WALL=$((T1-T0))
RTT_MEAS=$(grep -oE 'RTT_median_ms=[0-9.]+' "$RES/rtt.txt" | head -1 | cut -d= -f2)
NGATE=$(wc -l < "$RES/gate.txt" 2>/dev/null || echo 0)
PEAKMAX=$MAX0; [ "$MAX1" -gt "$PEAKMAX" ] 2>/dev/null && PEAKMAX=$MAX1
SHB=$(python3 -c "import json;print(json.load(open('$RES/shaper_stats.json'))['bytes'])" 2>/dev/null)
{
  echo "CELL=$CELL RATE=${RATE}mbit RTT=$RTT CONC=$CONC NLAYERS=$NLAYERS"
  echo "rc=$RC bb_wall_s=$BB_WALL outer_wall_s=$((T1-T0))"
  echo "peak_party0_kb=$MAX0 peak_party1_kb=$MAX1 peak_max_kb=$PEAKMAX"
  echo "rtt_meas_ms=${RTT_MEAS:-NA} gate_values=$NGATE shaped_bytes=${SHB:-NA}"
} | tee "$RES/RESULT.txt"
echo BB_CELL_DONE
exit "$RC"
