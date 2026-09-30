#!/bin/bash
#SBATCH --job-name=llkeybuf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/llkeybuf-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/llkeybuf-%j.err

# Source introspection (o0 image): readFile, the Buf classes' read/ownership contract, memBuf branches
# of init(), FileBuf. Baseline object: readFile() of server.dat, 94.3% of a 22.72 GiB peak.
# Nothing measured or published.
set -uo pipefail

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

SIF="${SIF:-/scratch/tmp/j_wred02/snni_campaign/llama/sif/llama-online-o0_default.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

apptainer exec "$SIF" bash -c '
R=/opt/EzPC/GPU-MPC/ext/sytorch

echo "=== where readFile is defined ==="
grep -rn "readFile" $R --include=*.h --include=*.cpp --include=*.impl 2>/dev/null | head -20

echo
echo "=== the buffer classes: who reads, who owns ==="
grep -rn "struct Buf\|class Buf\|struct .*Buf *[:{]\|class .*Buf *[:{]" $R --include=*.h 2>/dev/null | head -20

echo
echo "=== the Buf interface, verbatim ==="
for f in $(grep -rl "virtual.*read" $R --include=*.h 2>/dev/null | head -4); do
    echo "--- $f"
    grep -n -B4 -A20 "virtual.*read" "$f" | head -50
done

echo
echo "=== where memBuf decides, and what it builds ==="
grep -rn "memBuf" $R --include=*.h --include=*.cpp 2>/dev/null | head -20

echo
echo "=== init(): what the two branches build ==="
sed -n "85,130p" $R/include/sytorch/backend/llama_base.h

echo
echo "=== FileBuf, verbatim: does read(int) own its memory ==="
sed -n "96,152p" $R/ext/llama/include/llama/comms.h

echo
echo "=== does anything keep a pointer into the key buffer ==="
grep -rn "keyBuf->read(" $R --include=*.h --include=*.cpp 2>/dev/null | grep -v "char \*buf\|(char \*)" | head -12
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
