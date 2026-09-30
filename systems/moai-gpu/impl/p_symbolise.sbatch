#!/bin/bash
#SBATCH --job-name=mgsym
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/mgsym-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/mgsym-%j.err

# Resolves the two unsymbolised offsets holding 84% of the host peak directly against the binary.
# Introspection only; nothing measured or published.
set -uo pipefail
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

SIF="${SIF:-/scratch/tmp/j_wred02/snni_campaign/moai-gpu/sif/moai-gpu-s0_default.sif}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

apptainer exec "$SIF" bash -c '
BIN=/moai/build/test
echo "=== is the binary there, and does it carry debug info ==="
ls -la $BIN
command -v addr2line >/dev/null && echo "addr2line: yes" || echo "addr2line: NO -- that alone explains it"
command -v nm >/dev/null && echo "nm: yes" || echo "nm: no"
readelf -S $BIN 2>/dev/null | grep -c debug_info | sed "s/^/debug_info sections: /"

echo
echo "=== the two sites that hold 84% of the host peak ==="
for off in 0xf8fae 0xd71ab 0x47fe6 0x47c56 0x132311; do
    printf "%-10s " "$off"
    addr2line -e $BIN -f -C -i "$off" 2>&1 | tr "\n" " | "
    echo
done
'
RC=$?
echo "PROBE_EXIT_CODE=$RC"
exit "$RC"
