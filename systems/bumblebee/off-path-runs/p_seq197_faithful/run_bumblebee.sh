#!/bin/bash
# BumbleBee, inside the container. Two parties, one node, brpc over loopback. Both parties are the
# SAME binary/name/argv except --rank, so nothing tells them apart; this script WRITES ITS PARTY PIDS
# and the harness reads them. Deliberately thin: measurement decisions belong to campaign.env.
set -uo pipefail

RESULTS="${BB_RESULTS:?need BB_RESULTS}"
# op_bench writes the gate observable here directly; exported so the path stays a harness decision.
export BB_GATE_FILE="$RESULTS/gate.txt"
BIN=/opt/bumblebee/op_bench
N_LAYERS="${BB_N_LAYERS:?need BB_N_LAYERS}"
SEQ_LEN="${BB_SEQ_LEN:?need BB_SEQ_LEN}"
MAX_CONC="${BB_MAX_CONCURRENCY:?need BB_MAX_CONCURRENCY}"

mkdir -p "$RESULTS"
rm -f "$RESULTS"/party0.pid "$RESULTS"/party1.pid "$RESULTS"/gate.txt
md5sum "$BIN" | tee "$RESULTS/binary_md5.txt"

echo "=== op_bench rank 0 + rank 1, n_layers=$N_LAYERS seq_len=$SEQ_LEN max_concurrency=$MAX_CONC"
T0=$(date +%s)

# `env` execs the binary in its own process, so $! is the payload's pid and not a wrapper's.
env BB_N_LAYERS="$N_LAYERS" BB_SEQ_LEN="$SEQ_LEN" BB_MAX_CONCURRENCY="$MAX_CONC" \
    "$BIN" --rank 0 \
    > "$RESULTS/party0_stdout.log" 2> "$RESULTS/party0_markers.txt" &
P0=$!
echo "$P0" > "$RESULTS/party0.pid"

# Rank 0 binds 127.0.0.1:9530 and rank 1 binds :9531; brpc's connect retries, but starting them in
# the same instant loses the race often enough to be worth two seconds.
sleep 2

env BB_N_LAYERS="$N_LAYERS" BB_SEQ_LEN="$SEQ_LEN" BB_MAX_CONCURRENCY="$MAX_CONC" \
    "$BIN" --rank 1 \
    > "$RESULTS/party1_stdout.log" 2> "$RESULTS/party1_markers.txt" &
P1=$!
echo "$P1" > "$RESULTS/party1.pid"

wait "$P0"; RC0=$?
wait "$P1"; RC1=$?
echo "=== party0 rc=$RC0  party1 rc=$RC1  wall=$(( $(date +%s) - T0 ))s ==="
echo "$RC0" > "$RESULTS/exit_p0.txt"
echo "$RC1" > "$RESULTS/exit_p1.txt"

# THE SETUP MARKER: a binary predating the deterministic-input build looks identical until the gate
# cannot be compared.
grep -h '^SEEDED|' "$RESULTS/party0_markers.txt" "$RESULTS/party1_markers.txt" \
    > "$RESULTS/seeded.txt" 2>/dev/null
if [ ! -s "$RESULTS/seeded.txt" ]; then
    echo "FAILED: no SEEDED| marker; this binary is not the gated build" >&2
    [ "$RC0" = 0 ] && RC0=5
fi

# THE GATE OBSERVABLE: the reconstructed output tensor, one value per line (exact T1 or numeric T2).
# Both parties reconstruct; rank 0 writes it via BB_GATE_FILE and this script only CHECKS it.
# Recovering it from stderr once lost four fifths (brpc logs to fd 2 mid-run, splicing newlines):
# measured 19834 of 98304 values with a normal exit and a plausible file.
NGATE=$(wc -l < "$RESULTS/gate.txt" 2>/dev/null || echo 0)
EXPECTED=$(( SEQ_LEN * 768 ))
echo "gate values: $NGATE (expected $EXPECTED)"
if [ ! -s "$RESULTS/gate.txt" ] || [ "$NGATE" != "$EXPECTED" ]; then
    echo "FAILED: gate observable missing or short ($NGATE of $EXPECTED)" >&2
    [ "$RC0" = 0 ] && RC0=5
fi
if grep -q '^GATEFAIL|' "$RESULTS/party0_markers.txt" 2>/dev/null; then
    echo "FAILED: rank 0 could not write its gate file" >&2
    [ "$RC0" = 0 ] && RC0=5
fi
if ! grep -q '^GATE|layer1|rank=0|' "$RESULTS/party0_markers.txt" 2>/dev/null; then
    echo "FAILED: no GATE|layer1| marker; this binary predates the layer-1 gate" >&2
    [ "$RC0" = 0 ] && RC0=5
fi

# EVIDENCE BESIDE THE GATE, not the gate. GATESUM: one checksum per party per observation point; two
# cannot distinguish a reordering from a broken protocol, but a disagreement BETWEEN parties is a
# finding, and the `final` checksum documents the 12-layer wraparound behind gating layer 1.
grep -h '^GATESUM|' "$RESULTS/party0_markers.txt" "$RESULTS/party1_markers.txt" \
    > "$RESULTS/gatesum.txt" 2>/dev/null
for tag in layer1 final; do
    n=$(grep -h "^GATESUM|$tag|" "$RESULTS/gatesum.txt" 2>/dev/null | awk -F'|' '{print $5}' | sort -u | wc -l)
    [ "$n" -gt 1 ] && echo "WARNING: the two parties disagree on the $tag tensor"
done

# COMM: the exact data-independent companion invariant. Transcript byte counts are deterministic
# where the numeric observable has a tolerance, so the two check the same question two ways.
grep -h '^COMM|' "$RESULTS/party0_markers.txt" "$RESULTS/party1_markers.txt" \
    > "$RESULTS/comm.txt" 2>/dev/null
if [ ! -s "$RESULTS/comm.txt" ]; then
    echo "FAILED: no COMM| marker; the transcript invariant is missing" >&2
    [ "$RC0" = 0 ] && RC0=5
fi
cat "$RESULTS/comm.txt" 2>/dev/null

RC=$(( RC0 != 0 ? RC0 : RC1 ))
exit "$RC"
