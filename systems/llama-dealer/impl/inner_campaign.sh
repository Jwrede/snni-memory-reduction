#!/bin/bash
# LLAMA dealer, in-container step runner (lever baked into the image; this script only measures):
#   - poll_peak.sh at 100 ms, composition snapshot at each new maximum
#   - PMREC=1 resident recorder (own table subtracted): peak + decomposition from one run
#   - campaign output layout, read by tools/make_steps.py
# Usage: inner_campaign.sh <STEP>. Mounts: /work (key scratch), /results, /camp (tools).
set -uo pipefail
STEP="${1:?need STEP}"
S=/opt/EzPC/GPU-MPC/ext/sytorch
BIN=$S/build/bertbenchmark
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

[ -x "$BIN" ] || { echo "ERROR: $BIN missing -- the step image did not build"; exit 3; }

mkdir -p /results
{
  echo "step=$STEP"
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "role=dealer (party 1), measured alone"
  echo "image=${STEP_IMAGE:-unknown}"
  echo "image_digest=${STEP_IMAGE_DIGEST:-unknown}"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  echo "malloc_arena_max=$MALLOC_ARENA_MAX"
  echo "vram=NA (CPU system, single pool -- the degenerate case of the two-pool protocol)"
  echo "model=bert-base n_embd=768 n_layer=12 bw=51 n_seq=128 scale=12"
  echo "attrib_host=$([ -n "${PMREC:-}" ] && echo pmrec_pagemap || echo off)"
  echo "binary_md5=$(md5sum "$BIN" | awk '{print $1}')"
} > /results/RUN_META
cat /results/RUN_META

# The source state that produced this image, archived with the step.
(cd /opt/EzPC && git status --short > /results/src.status && git diff > /results/src.diff)
echo "src_diff_md5=$(md5sum /results/src.diff | awk '{print $1}')" >> /results/RUN_META

# --- run dealer (party 1) ---
cd /work || exit 2
rm -f server.dat client.dat
echo "=== dealer run (party 1)  pmrec=${PMREC:-off} ==="
T0=$(date +%s)
if [ -n "${PMREC:-}" ]; then
  SNNI_PM_TABLE=/results/pmtable SNNI_PM_MIN="${PM_MIN:-1048576}" LD_PRELOAD=/ar/pmrec.so \
    "$BIN" 1 > /results/stdout.log 2> /results/stderr.log &
else
  "$BIN" 1 > /results/stdout.log 2> /results/stderr.log &
fi
PID=$!
# The resident recorder adds one hash insert per allocation >= 1 MiB, nothing else; cheap enough
# for the clean run, unlike an allocator-replacing profiler.
SNNI_PM_TABLE="${PMREC:+/results/pmtable}" SNNI_PM_SAMPLE="${PMREC:+/ar/pmsample}" \
  bash /camp/poll_peak.sh "$PID" 100 /results/poll_dealer >/dev/null 2>&1 &
POLLPID=$!
RC=0; wait "$PID" || RC=$?
wait "$POLLPID" 2>/dev/null || true
T1=$(date +%s)

# --- correctness gate: the keys must be byte-identical across every step ---
# The PRG is deterministically seeded and levers change only WHEN memory is freed; a step whose
# keys differ changed the protocol, not its memory, and is void.
SZS=$(stat -c %s server.dat 2>/dev/null || echo 0)
SZC=$(stat -c %s client.dat 2>/dev/null || echo 0)
md5sum server.dat client.dat > /results/keys.md5 2>/dev/null || true
rm -f server.dat client.dat

echo $(( T1 - T0 )) > /results/wall_seconds.txt
PEAK_RSS=$(awk 'NR>1 && $2 ~ /^[0-9]+$/{if($2>m)m=$2}END{print m+0}' /results/poll_dealer/vmrss.log)
{
  echo "step=$STEP  exit=$RC  wall=$(( T1 - T0 ))s"
  echo "peak_host_kb=$PEAK_RSS"
  echo "peak_vram_kb=NA"
  echo "server_dat_B=$SZS  client_dat_B=$SZC"
} > /results/PEAK.txt
cat /results/PEAK.txt
cat /results/keys.md5 2>/dev/null || echo "no keyfiles (dealer failed?)"
exit "$RC"
