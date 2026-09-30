#!/bin/bash
#SBATCH --job-name=llscratch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/llscratch-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/llscratch-%j.err

# Source introspection after o2_activation_free (peak 1,201,520 kB): are the SlothGelu sites
# (173.0 MB) and SlothARS unfreed scratch or protocol floor; is anon:[heap] (189 MB) below-floor
# allocation or a missed mapping. Nothing measured or published.
set -uo pipefail

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

SIF="${SIF:-/scratch/tmp/j_wred02/snni_campaign/llama/sif/llama-online-o2_activation_free.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

apptainer exec "$SIF" bash -c '
A=$(find /opt -name api.cpp -path "*llama*" 2>/dev/null | head -1)
K=$(find /opt -name keypack.h -path "*llama*" 2>/dev/null | head -1)
echo "api.cpp = $A"
echo "keypack.h = $K"
[ -r "$A" ] || { echo "FAILED: api.cpp not in this image"; exit 3; }

echo
echo "=== the five SlothGelu sites in context, allocation through return ==="
sed -n "2690,2740p" "$A"

echo
echo "=== SlothARS around its allocation ==="
sed -n "4100,4140p" "$A"

echo
echo "=== SumOfSquare around its allocation ==="
sed -n "3260,3290p" "$A"

echo
echo "=== every delete/free inside SlothGelu, if any ==="
awk "/void SlothGelu|SlothGelu\\(int/,/^}/" "$A" | grep -n "new \|delete\|free(\|make_ars\|Key\|dealloc" | head -40

echo
echo "=== who calls SlothGelu, and how often per layer ==="
grep -n "SlothGelu(" "$A" | head -20

echo
echo "=== DPFETKeyPack: what it holds and whether it is freed ==="
sed -n "350,375p" "$K"
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
