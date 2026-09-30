#!/bin/bash
#SBATCH --job-name=moaigpu-det
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/det-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/det-%j.out
#
# Compares the gate observables of two unchanged runs; launches the baseline only on a match
# (SEEDS.md). Dependency afterany (failed guards stay visible).
set -uo pipefail
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/moai-gpu
R1=$B/results/det_check/run1
R2=$B/results/det_check/run2
OUT=$B/DETERMINISM.txt

fail() { echo "$*" | tee -a "$OUT" >&2; }

: > "$OUT"
echo "determinism check, $(date -u +%FT%TZ)" >> "$OUT"

BAD=0
for r in "$R1" "$R2"; do
    # program_exit_code.txt, not exit_status.txt (only the linkmode diagnostic writes the latter);
    # reading the wrong name once reported `rc=missing` and refused two perfect runs.
    rc=$(head -1 "$r/program_exit_code.txt" 2>/dev/null || echo missing)
    nm=$(grep -c '^MEM|' "$r/markers.log" 2>/dev/null || echo 0)
    wall=$(cat "$r/wall_seconds.txt" 2>/dev/null || echo 0)
    sd=$(grep -c '^SEEDED|' "$r/stderr.log" 2>/dev/null || echo 0)
    echo "$(basename "$r"): rc=$rc markers=$nm wall=${wall}s seeded=$sd" >> "$OUT"
    [ "$rc" = "0" ] || { fail "  REJECTED: rc=$rc"; BAD=1; }
    [ "$nm" -ge 16 ] 2>/dev/null || { fail "  REJECTED: $nm of 16 markers"; BAD=1; }
    [ "$wall" -ge 600 ] 2>/dev/null || { fail "  REJECTED: wall ${wall}s below 600"; BAD=1; }
    [ "$sd" -ge 1 ] 2>/dev/null || { fail "  REJECTED: no SEEDED marker, the seed did not take"; BAD=1; }
    [ -s "$r/gate.txt" ] || { fail "  REJECTED: empty gate.txt"; BAD=1; }
done

if [ "$BAD" -ne 0 ]; then
    fail "DETERMINISM: NOT ESTABLISHED, because a run did not measure what it claims."
    fail "The baseline is NOT launched. Fix the run, not the comparison."
    exit 2
fi

# The comparison itself. Byte for byte, which is what T1 means.
if cmp -s "$R1/gate.txt" "$R2/gate.txt"; then
    {
        echo "RESULT: byte-identical gate observables across two unchanged runs."
        echo "gate_tier T1 is CONFIRMED rather than intended."
        echo "--- the observable ---"
        grep -E '^GATE\|out\|' "$R1/gate.txt"
        grep -E '^GATE_NONZERO' "$R1/gate.txt"
    } >> "$OUT"
    cat "$OUT"

    # Also check the published line: a constant observable reproduces perfectly and proves nothing
    # (SHARK's first gate passed over 98,304 exact zeros).
    NZ=$(grep -oE '^GATE_NONZERO\|[0-9]+' "$R1/gate.txt" | cut -d'|' -f2)
    if [ -z "$NZ" ] || [ "$NZ" -eq 0 ] 2>/dev/null; then
        fail "REFUSING: the observable is constant zero. It reproduces and distinguishes nothing."
        exit 3
    fi
    echo "nonzero observable values: $NZ" >> "$OUT"

    echo
    echo "=== launching the baseline: three replicates, three channels, serial ==="
    exec bash "$C/harness/run_step.sh" moai-gpu s0_default
else
    {
        echo "RESULT: the two runs DISAGREE. gate_tier T1 does not hold."
        echo "This is not a failure of the check, it is what the check is for."
        echo "Next: quantify the disagreement and set a T2 tolerance from the MEASURED spread,"
        echo "the same shape BumbleBee's mechanism bound took. Do not widen a tolerance until it"
        echo "passes; derive it and then see whether it does."
        echo "--- first differing lines ---"
        diff "$R1/gate.txt" "$R2/gate.txt" | head -20
    } >> "$OUT"
    cat "$OUT"
    echo
    echo "The baseline is NOT launched, because a waterfall may not be published from a system"
    echo "whose gate cannot pass. SEEDS.md, and SHAFT-GPU is the reason it says so."
    exit 1
fi
