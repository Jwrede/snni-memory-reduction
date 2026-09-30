#!/bin/bash
#SBATCH --job-name=moaicpu-img
# Partition express: a normal-partition build waited ~10.5 h (14.08) for 4:32 of work.
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=90G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/img-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/img-%x-%j.err
#
#   sbatch --export=ALL,STEP=m0_default build_sif.sbatch
#
# From-scratch builds (SEAL + fork + project + >1 GB clone) need normal; a killed build leaves a
# partial image.
set -uo pipefail

B=/scratch/tmp/j_wred02/snni_campaign/moai-cpu
STEP="${STEP:?need STEP}"
OUT=$B/sif/moai-cpu-$STEP.sif
# Defaults to the baseline's def, overridable for a probe image (e.g. the pool census) with its own.
DEF="${DEF:-$B/moai-cpu.def}"

[ -r "$DEF" ] || { echo "no def at $DEF" >&2; exit 2; }
[ -r "$B/pin_seed_and_gate.py" ] || { echo "no patch script at $B/pin_seed_and_gate.py" >&2; exit 2; }
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists. Move it aside." >&2; exit 3; }
mkdir -p "$B/sif" || exit 2

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }
# Build sandbox on node-local tmp, not shared scratch. Extracting the >1GB localimage parent hung
# on scratch imgtmp (0 CPU, full hour) on 30-31.08 while the same build took 3.5min on 29.08. Only
# the finished OUT sif is written to scratch. SLURM sets TMPDIR under the node's /tmp (auto-cleaned).
LTMP="${TMPDIR:-/tmp}/apptainer-${SLURM_JOB_ID:-$$}"
mkdir -p "$LTMP/tmp" "$LTMP/cache" || exit 2
export APPTAINER_TMPDIR="$LTMP/tmp" APPTAINER_CACHEDIR="$LTMP/cache"
trap 'rm -rf "$LTMP"' EXIT

echo "=== building $OUT from $DEF on $(hostname), $(nproc) cores, at $(date -u +%FT%TZ)"
apptainer build "$OUT" "$DEF"
RC=$?
[ "$RC" = 0 ] || { echo "FAILED: apptainer build rc=$RC" >&2; exit "$RC"; }

echo "=== what the image says about itself"
apptainer exec "$OUT" cat /opt/moai_binary_md5.txt
apptainer exec "$OUT" cat /opt/moai_commit.txt
apptainer exec "$OUT" cat /opt/moai_alloc_imports.txt

# The knob has to work, not merely exist: a short run here proves it fires, not that it's correct.
echo "=== the depth knob announces itself"
apptainer exec --env MOAI_N_LAYERS=1 --env SNNI_SEED=0x5EED "$OUT" \
    bash -c 'cd /root/moai/data && timeout 20 /root/moai/build/test 2>&1 | grep -E "SNNI_PARAM\||WORKLOAD\||SEEDED\||MEM\|" | head -5'

echo "sha256 $(sha256sum "$OUT" | awk '{print $1}')"
echo "BUILD_OK $OUT"
