#!/bin/bash
#SBATCH --job-name=llscratch2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/llscratch2-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/llscratch2-%j.err

# Source introspection: SlothGelu's five arrays, leak or in-call scratch (the first probe's extraction
# may have stopped at a false closing brace). Expected: ~3.1 MB per call if freed, ~37.7 MB over 12
# layers if not; measured 34.6 MB per site. Nothing measured or published.
set -uo pipefail

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

SIF="${SIF:-/scratch/tmp/j_wred02/snni_campaign/llama/sif/llama-online-o2_activation_free.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

apptainer exec "$SIF" bash -c '
A=/opt/EzPC/GPU-MPC/ext/sytorch/ext/llama/api.cpp

echo "=== exact line of the definition ==="
grep -n "^void SlothGelu" "$A"

echo
echo "=== THE WHOLE FUNCTION, from its signature to the brace in column 0 that closes it ==="
START=$(grep -n "^void SlothGelu" "$A" | head -1 | cut -d: -f1)
awk -v s="$START" "NR>=s { print NR\": \"\$0; if (NR>s && \$0 ~ /^}/) exit }" "$A"

echo
echo "=== so: every delete[] between those two lines ==="
END=$(awk -v s="$START" "NR>s && /^}/ { print NR; exit }" "$A")
sed -n "${START},${END}p" "$A" | grep -c "delete\[\]" | sed "s/^/delete[] count: /"
sed -n "${START},${END}p" "$A" | grep -n "new GroupElement\|delete\[\]"

echo
echo "=== FOR CONTRAST, a neighbour that DOES release: SlothTR ==="
S2=$(grep -n "^void SlothTR" "$A" | head -1 | cut -d: -f1)
E2=$(awk -v s="$S2" "NR>s && /^}/ { print NR; exit }" "$A")
sed -n "${S2},${E2}p" "$A" | grep -n "new GroupElement\|delete\[\]"

echo
echo "=== the call site, which fixes size and the number of calls ==="
grep -rn "SlothGelu(" /opt/EzPC/GPU-MPC/ext/sytorch/include/ 2>/dev/null | head -10

echo
echo "=== and the model geometry the arithmetic above assumes ==="
grep -rn "3072\|intermediate\|n_embd\|seqlen\|128" /opt/EzPC/GPU-MPC/ext/sytorch/examples/bertbenchmark.cpp 2>/dev/null | head -12
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
