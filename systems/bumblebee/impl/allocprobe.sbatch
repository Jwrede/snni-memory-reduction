#!/bin/bash
#SBATCH --job-name=bballoc
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/bumblebee/allocprobe-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/bumblebee/allocprobe-%j.out
#
#   sbatch --export=ALL,STEP=b0_default allocprobe.sbatch
#
# Lists the allocation entry points the measured binary imports. Context: baseline full scan found
# 7.93 GB resident with zero recorded allocations (pmrec.so did not hook aligned_alloc, SEAL's path
# for 64-byte-multiple sizes).
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/bumblebee
SIF=$B/sif/bumblebee-$STEP.sif
BIN=/opt/bumblebee/op_bench

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 2; }

echo "=== image"
apptainer exec "$SIF" md5sum $BIN

echo
echo "=== allocation entry points this binary IMPORTS (U = undefined, resolved at load time)"
echo "    these are the ones an LD_PRELOAD recorder can interpose on"
apptainer exec "$SIF" bash -c "nm -D --with-symbol-versions $BIN 2>/dev/null \
    | grep -E ' U (malloc|calloc|realloc|free|aligned_alloc|memalign|posix_memalign|valloc|pvalloc|mmap|mmap64)\$|' \
    | grep -E ' U (malloc|calloc|realloc|free|aligned_alloc|memalign|posix_memalign|valloc|pvalloc|mmap|mmap64)' \
    | sort -u"

echo
echo "=== C++ allocation operators, mangled (Itanium ABI)"
echo "    _Znwm=new  _Znam=new[]  ...St11align_val_t=the C++17 over-aligned overloads"
apptainer exec "$SIF" bash -c "nm -D $BIN 2>/dev/null | grep -E '_Zn[wa][mj]' | sort -u"

echo
echo "=== does SEAL's aligned path exist in this build at all"
apptainer exec "$SIF" bash -c "strings $BIN | grep -c 'SEAL_USE_ALIGNED_ALLOC' || true"
apptainer exec "$SIF" bash -c "nm -C $BIN 2>/dev/null | grep -i 'MemoryPoolHeadMT::MemoryPoolHeadMT' | head -3"

echo
echo "=== which of them the RECORDER intercepts (from pmrec.c, for the record)"
grep -oE '^(void|int|void \*)[^(]*\b(malloc|calloc|realloc|free|posix_memalign|_Znwm|_Znam|_ZdlPv|_ZdaPv)\b' \
    "$C/attrib_resident/pmrec.c" | awk '{print $NF}' | sort -u | tr '\n' ' '
echo
echo ALLOCPROBE_DONE
