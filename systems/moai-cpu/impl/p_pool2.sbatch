#!/bin/bash
#SBATCH --job-name=moaicpu-pool2
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/probe-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/probe-%j.out

# Source introspection: is MemoryPoolThresholdMT (earlier campaign's snapshot, not stock SEAL 4.1)
# upstream code or that campaign's own allocator? Nothing measured or published.
set -uo pipefail

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

SIF="${SIF:-/scratch/tmp/j_wred02/snni_campaign/moai-cpu/sif/moai-cpu-m0_default.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

apptainer exec "$SIF" bash -c '
ROOT=/root/moai
S=$ROOT/thirdparty/SEAL-4.1-bs/native/src/seal
[ -d "$S" ] || { echo "FAILED: no SEAL source at $S"; exit 3; }

echo "=== every memory pool class this fork defines ==="
grep -n "class MemoryPool[A-Za-z]*" $S/util/mempool.h | head -20

echo
echo "=== and every MMProf ==="
grep -n "class MMProf[A-Za-z]*" $S/memorymanager.h | head -20

echo
echo "=== is MemoryPoolThresholdMT in this fork at all ==="
grep -rn "MemoryPoolThresholdMT" $S 2>/dev/null | head -5 || echo "(absent -- the earlier campaign added it)"

echo
echo "=== what the global profile actually is ==="
grep -n "MMProfGlobal\|GetPool()\|static.*MemoryManager" $S/memorymanager.h | head -15

echo
echo "=== the programs entry point ==="
sed -n "1,60p" $ROOT/test.cpp

echo
echo "=== where the measured run begins, i.e. the one marker ==="
grep -rn "MEM|run_begin" $ROOT/include $ROOT/test.cpp 2>/dev/null | head -5
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
