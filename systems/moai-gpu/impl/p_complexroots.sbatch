#!/bin/bash
#SBATCH --job-name=mgroots
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/mgroots-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/mgroots-%j.err

# Resolves the two unsymbolised offsets holding 84% of the host peak directly against the binary.
# Introspection only; nothing measured or published.
set -uo pipefail
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

SIF="${SIF:-/scratch/tmp/j_wred02/snni_campaign/moai-gpu/sif/moai-gpu-s0_default.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

apptainer exec "$SIF" bash -c '
P=/moai/thirdparty/phantom-fhe
echo "=== ComplexRoots, verbatim ==="
sed -n "1,60p" $P/src/fft.cu 2>/dev/null
echo
echo "=== its header ==="
find $P -name "fft.cuh" -o -name "fft.h" | head -2 | while read f; do echo "--- $f"; sed -n "1,60p" "$f"; done
echo
echo "=== who holds one, and for how long ==="
grep -rn "ComplexRoots" $P/src $P/include 2>/dev/null | grep -v "fft.cu:" | head -12
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
