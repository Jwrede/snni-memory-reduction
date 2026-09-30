#!/bin/bash
# SHAFT default GPU smoke test: one config (N=12, seq=128) on a single GPU/MIG slice.
# Both CrypTen parties share the one visible device.
set -uo pipefail

export SHAFT_MAX_LENGTH="${SHAFT_MAX_LENGTH:-128}"
export SHAFT_N_LAYERS="${SHAFT_N_LAYERS:-12}"
export SHAFT_DEVICE="${SHAFT_DEVICE:-cuda}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
RESULTS_BASE="${1:-/results/shaft_gpu}"
RESULTS_DIR="$RESULTS_BASE/n${SHAFT_N_LAYERS}_seq${SHAFT_MAX_LENGTH}"
mkdir -p "$RESULTS_DIR"
export RESULTS_DIR

echo "=== SHAFT GPU default smoke test ==="
echo "device=$SHAFT_DEVICE layers=$SHAFT_N_LAYERS seq=$SHAFT_MAX_LENGTH threads=$OMP_NUM_THREADS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python3 -c "import torch;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),
      'name',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')" || true
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || echo "nvidia-smi n/a"

rm -f "$RESULTS_DIR"/party*.pid

python3 /root/shaft/"${SHAFT_SCRIPT:-bert_instrumented_gpu.py}" \
    1>"$RESULTS_DIR/stdout.log" \
    2>"$RESULTS_DIR/all_markers.txt"
RC=$?

echo "--- exit code: $RC ---"
echo "=== stdout ==="; cat "$RESULTS_DIR/stdout.log" 2>/dev/null
echo "=== GPU peak markers ==="; grep -E '^GPU(PEAK|INFO)\|' "$RESULTS_DIR/all_markers.txt" 2>/dev/null
echo "=== any FATAL/Traceback ==="; grep -iE 'FATAL|Traceback|Error|CUDA' "$RESULTS_DIR/all_markers.txt" 2>/dev/null | head -20

# Split markers by party for later analysis
grep '^MEM|0|' "$RESULTS_DIR/all_markers.txt" > "$RESULTS_DIR/party0_markers.txt" 2>/dev/null || true
grep '^MEM|1|' "$RESULTS_DIR/all_markers.txt" > "$RESULTS_DIR/party1_markers.txt" 2>/dev/null || true
exit $RC
