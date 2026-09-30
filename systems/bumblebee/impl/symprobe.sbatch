#!/bin/bash
#SBATCH --job-name=bbsym
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/bumblebee/symprobe-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/bumblebee/symprobe-%j.out
#
#   sbatch --export=ALL,STEP=b0_default,ADDRS='0xf82e4c 0x69106f ...' symprobe.sbatch
#
# Resolves addresses with the correct offset and with tools/symbolise.py's formula (assumes
# p_vaddr == p_offset, false for this binary; named a 6.1 GB object as a libgcc unwinder symbol),
# addr2line -i for the inline chain.
set -uo pipefail

STEP="${STEP:?need STEP}"
ADDRS="${ADDRS:?need ADDRS, space-separated hex virtual addresses}"
BAD="${BAD:-}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/bumblebee
SIF=$B/sif/bumblebee-$STEP.sif
BIN=/opt/bumblebee/op_bench

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 2; }

echo "=== image"
apptainer exec "$SIF" md5sum $BIN
echo
echo "=== has this binary real debug info at all"
apptainer exec "$SIF" bash -c "readelf -S $BIN | grep -E 'debug_(info|line)' | head -4"
echo
echo "=== ELF segment layout (why the offset formula matters)"
apptainer exec "$SIF" bash -c "readelf -lW $BIN | awk '/LOAD/{print}'"
echo
echo "=== CORRECT offsets (addr - load_base), full inline chain"
for a in $ADDRS; do
    echo "--- $a"
    apptainer exec "$SIF" addr2line -f -C -i -e $BIN "$a"
done
if [ -n "$BAD" ]; then
    echo
    echo "=== the offsets the shared symboliser actually used, for comparison"
    for a in $BAD; do
        echo "--- $a"
        apptainer exec "$SIF" addr2line -f -C -i -e $BIN "$a"
    done
fi
echo SYMPROBE_DONE
