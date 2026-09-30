#!/bin/bash
#SBATCH --job-name=sharkdet
#SBATCH --partition=zen2-128C-496G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/shark/detcheck-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/shark/detcheck-%j.out
#
#   sbatch --export=ALL,STEP=s0_default,LAYERS=1 detcheck.sbatch
#
# Determinism pre-check (two unchanged runs, PROCEDURE.md) at 1 layer instead of 12. Checks the
# dealer's pinned PRNG seed, srand(0x5EED), and uninitialised spans (recycled heap bytes).
# Recorder preloaded.
set -uo pipefail

STEP="${STEP:?need STEP}"
LAYERS="${LAYERS:-1}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/shark
AR=$C/attrib_resident
SIF=$B/sif/shark-$STEP.sif
OUT=$B/detcheck/$STEP-L$LAYERS

[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT" || exit 2

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
# 16, matching the line it pre-checks (campaign decision 2026-08-03): the check must run at the
# operating point the line will run at, because thread count can be part of the seed.
export OMP_NUM_THREADS=16

for ROUND in A B; do
    RES=$OUT/round$ROUND
    WORK=/scratch/tmp/j_wred02/shark_work/det_${STEP}_$ROUND
    rm -rf "$WORK"; mkdir -p "$RES" "$WORK" || exit 2
    echo "=== round $ROUND (SHARK_N_LAYERS=$LAYERS) ==="
    INNER="export LD_PRELOAD=/ar/pmrec.so SNNI_PM_TABLE=$RES/pmtable_%d SNNI_PM_MIN=65536;"
    INNER="$INNER export SHARK_RESULTS=$RES SHARK_WORK=$WORK"
    INNER="$INNER SHARK_N_LAYERS=$LAYERS SHARK_SEQ_LEN=128 OMP_NUM_THREADS=$OMP_NUM_THREADS"
    INNER="$INNER; exec bash $B/run_shark_measured.sh"
    apptainer exec \
        --bind "$RES":"$RES" --bind "$WORK":"$WORK" \
        --bind "$B/run_shark_measured.sh":"$B/run_shark_measured.sh" \
        --bind "$AR":/ar \
        "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log"
    echo "round $ROUND rc=$?"
    rm -rf "$WORK"
    rm -f "$RES"/pmtable_*
done

echo
echo "=== gate observables"
cat "$OUT/roundA/gate.txt" 2>/dev/null
echo "---"
cat "$OUT/roundB/gate.txt" 2>/dev/null
echo

RC=0
if [ ! -s "$OUT/roundA/gate.txt" ] || [ ! -s "$OUT/roundB/gate.txt" ]; then
    echo "FAILED: at least one round produced no gate observable"
    RC=1
elif diff -q "$OUT/roundA/gate.txt" "$OUT/roundB/gate.txt" >/dev/null; then
    echo "DETERMINISM OK: gate observables byte-identical across two unchanged runs"
else
    echo "NONDETERMINISTIC: the two unchanged runs disagree. This system has no gate."
    diff "$OUT/roundA/gate.txt" "$OUT/roundB/gate.txt" | head -20
    RC=1
fi

# The full output vectors, compared element by element, so a disagreement has a SIZE and not only
# a verdict. A bare FAIL cannot distinguish a reordering from a broken protocol.
for P in 0 1; do
    A=$OUT/roundA/output_party$P.txt
    Bf=$OUT/roundB/output_party$P.txt
    if [ -s "$A" ] && [ -s "$Bf" ]; then
        n=$(paste "$A" "$Bf" | awk '$1!=$2{c++} END{print c+0}')
        tot=$(wc -l < "$A")
        echo "party$P: $n of $tot output elements differ between the two runs"
    fi
done

# The two parties must also agree with EACH OTHER: both reconstruct the same vector, so a
# difference there is a protocol fault rather than a determinism question.
if [ -s "$OUT/roundA/output_party0.txt" ] && [ -s "$OUT/roundA/output_party1.txt" ]; then
    n=$(paste "$OUT/roundA/output_party0.txt" "$OUT/roundA/output_party1.txt" \
        | awk '$1!=$2{c++} END{print c+0}')
    echo "round A: $n output elements differ BETWEEN party 0 and party 1 (expected 0)"
fi

exit $RC
