#!/bin/bash
#SBATCH --job-name=moaicpu-pool
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/probe-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/probe-%j.out

# Source introspection: 99.3% of the peak is MemoryPoolHeadMT::get() (SEAL pool). Checks which of the
# earlier campaign's mechanisms (phase-split keys, MMProfThreshold, chunked bootstrap;
# CANONICAL_RESULTS.md, 72.3 GiB) this pinned image can carry. Nothing measured or published.
set -uo pipefail

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

SIF="${SIF:-/scratch/tmp/j_wred02/snni_campaign/moai-cpu/sif/moai-cpu-m0_default.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

apptainer exec "$SIF" bash -c '
echo "=== where the measured source lives ==="
# /root/moai (SEAL header confirms it); an empty ROOT here silently greps the wrong tree instead.
ROOT=/root/moai
[ -d "$ROOT" ] || { echo "FAILED: no source tree at $ROOT in this image"; exit 3; }
echo "root = $ROOT"
ls "$ROOT" | head -20

echo
echo "=== does this build touch SEALs MemoryManager AT ALL ==="
grep -rn "MemoryManager\|MMProf\|SwitchProfile\|MemoryPoolHandle" $ROOT 2>/dev/null | head -20

echo
echo "=== what SEAL offers: the profile classes in this version of the header ==="
H=$(find / -name "memorymanager.h" -path "*seal*" 2>/dev/null | head -1)
echo "header = $H"
grep -n "class MMProf\|SwitchProfile\|GetPool\|MMProfThreshold\|MMProfFixed\|MMProfNew\|MMProfGlobal" "$H" 2>/dev/null | head -25

echo
echo "=== the Galois keys, since phase-split keys is the second donor mechanism ==="
grep -rn "create_galois_keys\|GaloisKeys\|galois_key\|gal_keys" $ROOT 2>/dev/null | head -15

echo
echo "=== and the bootstrap loop, the third ==="
grep -rn "bootstrap\|Bootstrap" $ROOT 2>/dev/null | head -15

echo
echo "=== phase boundaries that a marker COULD attach to, if a probe needs one ==="
grep -rn "MEM|run_begin\|std::cout << \"\\[" $ROOT 2>/dev/null | head -15
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
