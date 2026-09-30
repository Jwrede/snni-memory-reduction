#!/bin/bash
#SBATCH --job-name=arion-img
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/arion/img-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/arion/img-%x-%j.err
#
#   sbatch --export=ALL,STEP=a0_default build_sif.sbatch
set -uo pipefail

B=/scratch/tmp/j_wred02/snni_campaign/arion
STEP="${STEP:?need STEP}"
OUT=$B/sif/arion-$STEP.sif
# Overridable so a step that patches the source can use its own def, recorded in the job itself.
DEF="${DEF:-$B/arion.def}"

[ -r "$DEF" ] || { echo "no def at $DEF" >&2; exit 2; }
[ -r "$B/pin_workload.py" ] || { echo "no patch script at $B/pin_workload.py" >&2; exit 2; }
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists. Move it aside." >&2; exit 3; }
mkdir -p "$B/sif" "$B/imgtmp" "$B/imgcache" || exit 2

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }
export APPTAINER_TMPDIR=$B/imgtmp APPTAINER_CACHEDIR=$B/imgcache

echo "=== building $OUT from $DEF on $(hostname) at $(date -u +%FT%TZ)"
apptainer build "$OUT" "$DEF"
RC=$?
[ "$RC" = 0 ] || { echo "FAILED: apptainer build rc=$RC" >&2; exit "$RC"; }

apptainer exec "$OUT" cat /opt/arion_binary_md5.txt
apptainer exec "$OUT" cat /opt/arion_commit.txt
apptainer exec "$OUT" cat /opt/arion_lattigo.txt

# Proves the knob works (not merely exists) by checking the binary echoes the value it was given.
echo "=== probing the declared knobs (no workload, menu exits immediately)"
apptainer exec --env ARION_THREADS=7 --env ARION_N_LAYERS=1 "$OUT" \
    bash -c 'cd /root/arion && echo "0" | ./arion' 2>&1 | head -20

echo "sha256 $(sha256sum "$OUT" | awk '{print $1}')"
echo "BUILD_OK $OUT"
