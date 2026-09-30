#!/bin/bash
#SBATCH --job-name=bb197nr
#SBATCH --partition=normal
#SBATCH --exclusive --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/bb197norec/logs/%x-%j.out
# seq-197 transfer probe of the faithful line without the object recorder (recorder + freeze crash
# party 1 on the b4 tree at seq 197, ../bb197diag). Peaks only: VmHWM per party polled, max kept.
set -uo pipefail
module load Apptainer/1.5.0
: "${TAG:?}"; : "${SIF:?}"; : "${CONC:?}"
WD=/scratch/tmp/j_wred02/bb197norec
R=$WD/$TAG; rm -rf $R; mkdir -p $R
echo "tag=$TAG sif=$SIF conc=$CONC seq=197 layers=12 host=$(hostname) date=$(date -u +%FT%TZ)" > $R/META.txt
sha256sum $SIF | cut -c1-16 >> $R/META.txt
T0=$(date +%s)
apptainer exec --bind $R:$R --bind $WD:$WD $SIF bash -c "export BB_RESULTS=$R BB_N_LAYERS=12 BB_SEQ_LEN=197 BB_MAX_CONCURRENCY=$CONC OMP_NUM_THREADS=16 ${TMO:+BB_RECV_TIMEOUT_MS=$TMO}; exec bash $WD/run_bumblebee.sh" > $R/outer.log 2>&1 &
APP=$!
for _ in $(seq 1 6000); do [ -s $R/party0.pid ] && [ -s $R/party1.pid ] && break; kill -0 $APP 2>/dev/null || break; sleep 0.1; done
P0=$(cat $R/party0.pid 2>/dev/null); P1=$(cat $R/party1.pid 2>/dev/null)
hwm() { awk "/^VmHWM:/{print \$2}" /proc/$1/status 2>/dev/null; }
M0=0; M1=0
while kill -0 $APP 2>/dev/null; do h0=$(hwm $P0); h1=$(hwm $P1); [ -n "$h0" ] && [ "$h0" -gt $M0 ] && M0=$h0; [ -n "$h1" ] && [ "$h1" -gt $M1 ] && M1=$h1; sleep 0.1; done
wait $APP; RC=$?
W=$(grep -oE "wall=[0-9]+s" $R/outer.log | tail -1 | tr -dc 0-9)
PM=$M0; [ $M1 -gt $PM ] && PM=$M1
echo "rc=$RC wall_s=$W peak_party0_kb=$M0 peak_party1_kb=$M1 peak_max_kb=$PM" | tee $R/RESULT.txt
exit $RC
