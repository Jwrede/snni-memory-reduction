#!/bin/bash
#SBATCH --job-name=sharkfloor
#SBATCH --partition=zen2-128C-496G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/shark/floor-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/shark/floor-%j.out
#
#   sbatch --export=ALL,STEP=s0_default floor_bench.sbatch
#
# Runs each candidate target operation alone at BERT-base dimensions (same image and machine type
# as the line) to check target_host_kb.
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/shark
SIF=$B/sif/shark-$STEP.sif
OUT=$B/floor_bench/$STEP
WORK=/scratch/tmp/j_wred02/shark_work/floor_$STEP

[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 2; }
rm -rf "$WORK"; mkdir -p "$WORK" "$OUT" || exit 2

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

export OMP_NUM_THREADS=4

apptainer exec --bind "$OUT":"$OUT" --bind "$WORK":"$WORK" "$SIF" bash -c "
set -uo pipefail
cd '$WORK' || exit 9
export OMP_NUM_THREADS=$OMP_NUM_THREADS
BIN=/root/shark/build/benchmark-shark_floor

echo '=== dealer (party 2): writes the preprocessing for every benched operation'
\"\$BIN\" 2 > '$OUT/floor_dealer.out' 2> '$OUT/floor_dealer.err'
echo \"dealer rc=\$?\"
stat -c '%n %s' server.dat client.dat

echo '=== online parties, oneShot OFF so each operation shows its own working set'
export SHARK_ONESHOT=0
\"\$BIN\" 0 > '$OUT/floor_party0.out' 2> '$OUT/floor_party0.err' &
P0=\$!
sleep 3
\"\$BIN\" 1 > '$OUT/floor_party1.out' 2> '$OUT/floor_party1.err' &
P1=\$!
wait \$P0; echo \"party0 rc=\$?\"
wait \$P1; echo \"party1 rc=\$?\"
" 2>&1 | tee "$OUT/driver.log"

echo
echo "=== FLOOR lines (party 0 = server, party 1 = client)"
grep -h '^FLOOR' "$OUT"/floor_party0.err "$OUT"/floor_party1.err 2>/dev/null | sort

echo
echo "=== the largest single operation, by key bytes, on party 0"
awk -F'|' '/^FLOOR\|/ {
        split($5, a, "="); kb = a[2] + 0;
        if (kb > best) { best = kb; line = $0 }
    } END { if (line != "") printf "%s\n  -> %.1f MB of key material\n", line, best/1e6 }' \
    "$OUT/floor_party0.err" 2>/dev/null

rm -rf "$WORK"
grep -q 'FLOORDONE' "$OUT/floor_party0.err" 2>/dev/null \
    && echo "FLOOR_OK" || { echo "FAILED: floor bench did not run to completion"; exit 1; }
