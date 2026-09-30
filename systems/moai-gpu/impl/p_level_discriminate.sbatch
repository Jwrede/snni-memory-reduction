#!/bin/bash
#SBATCH --job-name=moaigpu-leveldisc
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/leveldisc-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/leveldisc-%j.out
#
# Discrimination test of the structural gate (2026-08-03): image with one fewer mod-switch
# (test_single_layer.cuh:868) against the unbroken trace. PASS = traces differ.
set -uo pipefail
B=/scratch/tmp/j_wred02/snni_campaign/moai-gpu
STEP="${STEP:-p_level_discriminate}"
OUT=$B/results/$STEP
SIF=$B/sif/moai-gpu-$STEP.sif
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT"
module purge; module load Apptainer/1.5.0

T0=$(date +%s)
set +e
apptainer exec --nv --writable-tmpfs --pwd /moai/build \
    "$SIF" bash -lc 'OMP_NUM_THREADS=8 SNNI_SEED=0x5EED ./test' \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
RC=$?
set -e
T1=$(date +%s)
echo "PROGRAM_EXIT_CODE=$RC"
echo "wall_s=$((T1-T0))"
[ "$RC" -gt 128 ] && echo "died on signal $((RC-128))"

# The trace: the program's own chain-index lines, in order.
grep -E "Modulus chain index" "$OUT/stdout.log" > "$OUT/leveltrace.txt" || true
echo "trace_lines=$(wc -l < "$OUT/leveltrace.txt")"

# Unbroken reference: a clean det_check run of the current image. Globs directly with `set +e`
# rather than `ls glob | head -1`, whose unmatched-glob exit under pipefail once killed the script
# silently right before its verdict (job 45755949, 2026-08-04).
REF=""
set +e
for _c in "$B"/results/det_check/*/stdout.log "$B"/results/det_check/*/*/stdout.log; do
    [ -s "$_c" ] && { REF="$_c"; break; }
done
set -e
if [ -z "$REF" ]; then
    echo "NO REFERENCE stdout under results/det_check; cannot compare" >&2
    exit 3
fi
grep -E "Modulus chain index" "$REF" > "$OUT/leveltrace_ref.txt" || true
echo "ref=$REF ref_lines=$(wc -l < "$OUT/leveltrace_ref.txt")"
md5sum "$OUT/leveltrace.txt" "$OUT/leveltrace_ref.txt"

if [ ! -s "$OUT/leveltrace.txt" ]; then
    echo "VERDICT: NO TRACE from the broken run (rc=$RC) -- the probe did not complete far"
    echo "enough to emit levels; that is a crash result, not a discrimination result."
    exit 4
fi
# A probe ending early still answers the question if it diverged first; compares the common prefix.
NP=$(wc -l < "$OUT/leveltrace.txt"); NR_=$(wc -l < "$OUT/leveltrace_ref.txt")
if [ "$NP" -lt "$NR_" ]; then
    echo "NOTE: the probe emitted $NP of the reference's $NR_ trace lines (rc=$RC); comparing the"
    echo "      common prefix. A divergence inside it is still a divergence."
    head -n "$NP" "$OUT/leveltrace_ref.txt" > "$OUT/leveltrace_ref_prefix.txt"
    mv "$OUT/leveltrace_ref_prefix.txt" "$OUT/leveltrace_ref.txt"
fi
if diff -q "$OUT/leveltrace.txt" "$OUT/leveltrace_ref.txt" >/dev/null; then
    echo "VERDICT: FAIL -- the modulus-chain trace did NOT move under a structural change."
    echo "The structural gate licenses nothing and MOAI-GPU has no gate."
    exit 5
else
    echo "VERDICT: PASS -- the trace moved under a structural change. First divergence:"
    # `|| true`: diff's exit 1 (the success case here) under set -e once reported a PASS as FAILED.
    diff "$OUT/leveltrace_ref.txt" "$OUT/leveltrace.txt" | head -12 || true
    exit 0
fi
