#!/bin/bash
#SBATCH --job-name=moaicpu-probe
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/probe-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/probe-%j.out
#
#   sbatch source_probe.sbatch
#
# Reconnaissance on the existing image (job, not login node): live test call, run depth, randomness
# source, image self-rebuild.
set -uo pipefail

SRC=${SRC:-/scratch/tmp/j_wred02/moai_cpu/moai-full-bert.sif}
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
[ -r "$SRC" ] || { echo "FAILED: no image at $SRC"; exit 1; }
echo "=== probing $SRC"
sha256sum "$SRC" | head -1

apptainer exec "$SRC" bash -lc '
echo "########## 1. which test call is LIVE in test.cpp"
sed -n "1,200p" /root/moai/test.cpp | grep -nE "^\s*[a-zA-Z_]+_test\(\);" || echo "(none uncommented at top level)"

echo
echo "########## 2. layer depth"
grep -rnE "for\s*\(int\s+(i|layer|l)\s*=\s*0;\s*(i|layer|l)\s*<" /root/moai/include/test/test_full_scheme.hpp | head -10
grep -rn "num_layer\|NUM_LAYER" /root/moai/include/test/test_full_scheme.hpp | head -5

echo
echo "########## 3. randomness"
grep -rnE "KeyGenerator|random_device|mt19937|srand|rand\(\)|prng|PRNG|random_generator" \
    /root/moai/include/test/test_full_scheme.hpp | head -15
echo "--- does the bundled SEAL expose a seedable PRNG factory"
grep -rl "Blake2xbPRNGFactory" /root/moai/thirdparty 2>/dev/null | head -3

echo
echo "########## 4. can this image rebuild itself"
ls -d /root/moai/build 2>/dev/null && ls /root/moai/build | head -10
which g++ cmake make 2>/dev/null
g++ --version 2>/dev/null | head -1
ls -la /root/moai/build/test 2>/dev/null && md5sum /root/moai/build/test 2>/dev/null

echo
echo "########## 5. what the binary already prints (markers, gate, MEM)"
strings -a /root/moai/build/test 2>/dev/null | grep -E "^(MEM\||GATE|SEEDED|WORKLOAD)" | head -10
echo "(nothing above means the binary carries no campaign markers yet)"

echo
echo "########## 6. the data the run reads"
ls /root/moai/data 2>/dev/null | head -5
du -sh /root/moai/data 2>/dev/null
'
echo "PROBE_DONE"
