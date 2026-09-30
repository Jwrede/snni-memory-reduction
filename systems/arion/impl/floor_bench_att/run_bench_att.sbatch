#!/bin/bash
#SBATCH --job-name=arion_floor_att
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/j_wred02/arion_target_probe/bench_att/bench_att-%j.out
# ARION QK^T floor micro-benchmark, T=1, one integrated set. Analogue of moai-cpu p_floor_bench_att.
module load Apptainer/1.5.0 2>/dev/null
cd /scratch/tmp/j_wred02/arion_target_probe/bench_att
echo "node=$(hostname) date=$(date -u +%FT%TZ) job=$SLURM_JOB_ID"
apptainer exec --bind /scratch /scratch/tmp/j_wred02/snni_campaign/arion/sif/arion-a4_activation_input_release_goheap.sif \
  env GOMAXPROCS=1 GOGC=100 ./bench_att
echo "rc=$?"
