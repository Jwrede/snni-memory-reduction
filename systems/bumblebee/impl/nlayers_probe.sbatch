#!/bin/bash
#SBATCH --job-name=bbnprobe
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/bumblebee/nprobe-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/bumblebee/nprobe-%j.out
#
#   sbatch --export=ALL,STEP=b0_default nlayers_probe.sbatch
#
# Cause of a 3.4x output-checksum disagreement between two unchanged 12-layer runs: (1) randomness
# at first order (needs SPU RNG pinning) or (2) ~1 ULP protocol error amplified by ring wraparound.
# Discriminator: layer count N=1 vs N=12 via GATESUM. No poller or recorder.
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/bumblebee
SIF="${SIF:-$B/sif/bumblebee-$STEP.sif}"
OUT=$B/nprobe/$STEP
BIN=/opt/bumblebee/op_bench

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT" || exit 2

run_once() {   # <n_layers> <tag>
    local n="$1" tag="$2"
    local d="$OUT/n${n}_$tag"
    mkdir -p "$d"
    apptainer exec --bind "$d":"$d" "$SIF" bash -c "
        export BB_N_LAYERS=$n BB_SEQ_LEN=128 BB_MAX_CONCURRENCY=16 OMP_NUM_THREADS=16
        $BIN --rank 0 > $d/p0.out 2> $d/p0.err &
        P0=\$!
        sleep 2
        $BIN --rank 1 > $d/p1.out 2> $d/p1.err &
        P1=\$!
        wait \$P0; echo \$? > $d/rc0
        wait \$P1; echo \$? > $d/rc1
    " > "$d/shell.log" 2>&1
    printf 'n=%s %-6s rc=%s/%s  ' "$n" "$tag" "$(cat "$d/rc0" 2>/dev/null)" "$(cat "$d/rc1" 2>/dev/null)"
    grep -h '^GATESUM|rank=0|' "$d/p0.err" 2>/dev/null | tail -1
}

echo "=== image $SIF"
apptainer exec "$SIF" md5sum $BIN

for n in 1 12; do
    for tag in a b; do
        run_once "$n" "$tag"
    done
done

echo
echo "=== verdict inputs"
for n in 1 12; do
    a=$(grep -h '^GATESUM|rank=0|' "$OUT/n${n}_a/p0.err" 2>/dev/null | tail -1 | awk -F'|' '{print $4}')
    b=$(grep -h '^GATESUM|rank=0|' "$OUT/n${n}_b/p0.err" 2>/dev/null | tail -1 | awk -F'|' '{print $4}')
    if [ -n "$a" ] && [ "$a" = "$b" ]; then
        echo "N=$n: IDENTICAL ($a)"
    else
        echo "N=$n: DIFFER ($a vs $b)"
    fi
done
echo NPROBE_DONE
