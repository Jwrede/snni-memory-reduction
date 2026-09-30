#!/bin/bash
#SBATCH --job-name=bolt-img
# Partition express: normal had a 10.5 h queue wait for an 8:43 build.
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=90G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/bolt/img-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/bolt/img-%x-%j.err
#
#   sbatch --export=ALL,STEP=s0_default build_sif.sbatch
#   sbatch --export=ALL,STEP=<step>,DEF=$B/bolt-<step>.def build_sif.sbatch   (a step's own def)
#
# Compiles SCI + vendored SEAL 4.1.1, downloads 780 MB of weights. A build killed at the time
# limit leaves a partial image.
set -uo pipefail

B=/scratch/tmp/j_wred02/snni_campaign/bolt
STEP="${STEP:?need STEP}"
OUT=$B/sif/bolt-$STEP.sif
DEF="${DEF:-$B/bolt.def}"

[ -r "$DEF" ] || { echo "no def at $DEF" >&2; exit 2; }
[ -r "$B/pin_seed_and_gate.py" ] || { echo "no patch script at $B/pin_seed_and_gate.py" >&2; exit 2; }
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists. Move it aside." >&2; exit 3; }
mkdir -p "$B/sif" "$B/imgtmp" "$B/imgcache" || exit 2

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }
export APPTAINER_TMPDIR=$B/imgtmp APPTAINER_CACHEDIR=$B/imgcache

echo "=== building $OUT from $DEF on $(hostname), $(nproc) cores, at $(date -u +%FT%TZ)"
apptainer build "$OUT" "$DEF"
RC=$?
[ "$RC" = 0 ] || { echo "FAILED: apptainer build rc=$RC" >&2; exit "$RC"; }

echo "=== what the image says about itself"
apptainer exec "$OUT" cat /opt/bolt_binary_md5.txt
apptainer exec "$OUT" cat /opt/bolt_ezpc_commit.txt
apptainer exec "$OUT" cat /opt/bolt_alloc_imports.txt

# Weights checked here (not at first measurement run) since they're what a rebuild can't reproduce if the download link dies.
N=$(apptainer exec "$OUT" bash -c 'ls /opt/bolt_data/quantize/sst-2/weights_txt/ | wc -l')
echo "weight files: $N"
[ "$N" -gt 100 ] || { echo "FAILED: weight set looks short ($N files)" >&2; exit 4; }

echo "sha256 $(sha256sum "$OUT" | awk '{print $1}')"
echo "BUILD_OK $OUT"
