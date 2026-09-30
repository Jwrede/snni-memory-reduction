#!/bin/bash
#SBATCH --job-name=pumacfg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/logs/pumacfg-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/logs/pumacfg-%j.err

# Copies the SPU runtime config out of the image, so a step can bind a modified copy via PUMA_CONFIG.
set -uo pipefail
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

B=/scratch/tmp/j_wred02/snni_campaign/puma
SIF="${SIF:-$B/sif/puma-s0_default.sif}"
OUT="${OUT:-$B/config/3pc.json}"
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF" >&2; exit 2; }

mkdir -p "$(dirname "$OUT")"
apptainer exec "$SIF" cat /opt/puma/examples/python/ml/flax_bert/3pc.json > "$OUT" || exit 3
[ -s "$OUT" ] || { echo "FAILED: extracted config is empty" >&2; exit 3; }
python3 -c "import json,sys; d=json.load(open('$OUT')); assert 'nodes' in d and 'devices' in d; print('nodes:', list(d['nodes'])); print('devices:', list(d['devices']))" || exit 4
echo "--- was ueber Nebenlaeufigkeit drinsteht ---"
grep -n -i "concurrency\|thread\|parallel" "$OUT" || echo "(kein Nebenlaeufigkeits-Schluessel gesetzt, also Default)"
md5sum "$OUT"
exit 0
