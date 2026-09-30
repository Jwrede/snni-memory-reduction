#!/bin/bash
#SBATCH --job-name=vit_floor
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/tmp/j_wred02/vit_floor/vit_floor-%j.out
set -uo pipefail
module load Apptainer 2>/dev/null
SIF=/scratch/tmp/j_wred02/snni_campaign/shaft-cpu/sif/shaft-cpu-s0_default.sif
cd /scratch/tmp/j_wred02/vit_floor
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "=== node: $(hostname) mem: $(free -g|awk '/Mem:/{print $2}')GB $(date -u) ==="
apptainer exec "$SIF" python3 vit_floor_bench.py
echo "=== done rc=$? ==="
