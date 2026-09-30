#!/bin/bash
# SHAFT-CPU: ONE configuration (N=12 layers, seq=128). The image's run.sh sweeps six layer counts
# in one process tree sharing one wall time and poll trace, so it cannot be a measured step.
set -uo pipefail

export SHAFT_MAX_LENGTH="${SHAFT_MAX_LENGTH:-128}"
export SHAFT_N_LAYERS="${SHAFT_N_LAYERS:-12}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:?the sbatch must pin the thread count}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

RESULTS_DIR="${1:?need results dir}"
mkdir -p "$RESULTS_DIR"
export RESULTS_DIR

echo "=== SHAFT-CPU ==="
echo "layers=$SHAFT_N_LAYERS seq=$SHAFT_MAX_LENGTH threads=$OMP_NUM_THREADS"
echo "levers=${SHAFT_LEVERS:-<none, baseline>}"
python3 -c "import torch,crypten,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)" || true

rm -f "$RESULTS_DIR"/party*.pid

# The poller descends from this shell to the python launcher; the parties it spawns write their
# own pids into RESULTS_DIR.
python3 /root/shaft/bert_instrumented_cpu.py \
    1>"$RESULTS_DIR/stdout.log" \
    2>"$RESULTS_DIR/all_markers.txt"
RC=$?

echo "--- exit code: $RC ---"
echo "=== stdout ==="; cat "$RESULTS_DIR/stdout.log" 2>/dev/null
echo "=== seed + lever markers ==="; grep -E '^(SEEDED|LEVER|CONFIG)\|' "$RESULTS_DIR/all_markers.txt" 2>/dev/null
echo "=== any FATAL/Traceback ==="; grep -iE 'FATAL|Traceback|Error' "$RESULTS_DIR/all_markers.txt" 2>/dev/null | head -20

grep '^MEM|0|' "$RESULTS_DIR/all_markers.txt" > "$RESULTS_DIR/party0_markers.txt" 2>/dev/null || true
grep '^MEM|1|' "$RESULTS_DIR/all_markers.txt" > "$RESULTS_DIR/party1_markers.txt" 2>/dev/null || true
exit $RC
