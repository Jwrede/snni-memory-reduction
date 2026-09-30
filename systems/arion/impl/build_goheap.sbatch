#!/bin/bash
#SBATCH --job-name=arion-goheap-img
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/arion/img-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/arion/img-%x-%j.err
#
# Builds the p_goheap census image: baseline source at the same commit (read from the published
# baseline image) plus the dormant heap-dump adapter.
set -euo pipefail
B=/scratch/tmp/j_wred02/snni_campaign/arion
#
# NAME/DEF OVERRIDABLE: the census image is per step (the adapter is compiled in), defaulting to
# the baseline's.
#
#   sbatch --export=ALL,NAME=a1_x_goheap,DEF=$B/arion-a1_x_goheap.def build_goheap.sbatch
NAME="${NAME:-p_goheap}"
OUT=$B/sif/arion-$NAME.sif
DEF="${DEF:-$B/arion-goheap.def}"
BASE=$B/sif/arion-a0_default.sif

[ -r "$BASE" ] || { echo "no baseline image at $BASE" >&2; exit 2; }
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists. Move it aside." >&2; exit 3; }
[ -r "$B/snni_heapdump.go" ] || { echo "no adapter source at $B/snni_heapdump.go" >&2; exit 2; }

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

apptainer exec "$BASE" cat /opt/arion_commit.txt > "$B/goheap_commit.txt"
echo "baseline commit: $(cat "$B/goheap_commit.txt")"
[ -s "$B/goheap_commit.txt" ] || { echo "could not read the baseline commit" >&2; exit 2; }

export APPTAINER_TMPDIR=$B/imgtmp APPTAINER_CACHEDIR=$B/imgcache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
[ -r "$DEF" ] || { echo "no def at $DEF" >&2; exit 2; }
apptainer build "$OUT" "$DEF"
sha256sum "$OUT"
echo "GOHEAP_IMAGE_OK"
