#!/bin/bash
# LLAMA online, in-container runner. Two protocol phases; only phase 2 is measured.
#   phase 1  the dealer alone writes server.dat (23.0 GB) and client.dat (23.9 GB)
#   phase 2  server and client run concurrently, each reading its own key file
# Party pids are written at the start of phase 2, so the poller never attaches during key generation
# (the dealer is the separate system llama-dealer).
set -uo pipefail

RESULTS="${LLAMA_RESULTS:?need LLAMA_RESULTS}"
WORK="${LLAMA_WORK:?need LLAMA_WORK}"
BIN=/opt/EzPC/GPU-MPC/ext/sytorch/build/bertbenchmark

cd "$WORK" || exit 9
mkdir -p "$RESULTS"
rm -f server.dat client.dat "$RESULTS"/party2.pid "$RESULTS"/party3.pid

md5sum "$BIN" | tee "$RESULTS/binary_md5.txt"

echo "=== phase 1: dealer (party 1), NOT measured ==="
T0=$(date +%s)
"$BIN" 1 > "$RESULTS/dealer.log" 2>&1
RCD=$?
echo "dealer rc=$RCD  wall=$(( $(date +%s) - T0 ))s"
ls -la server.dat client.dat 2>/dev/null | tee "$RESULTS/keyfiles.txt"
if [ "$RCD" != 0 ] || [ ! -s server.dat ] || [ ! -s client.dat ]; then
    echo "FAILED: key generation did not produce both key files; nothing to measure" >&2
    exit 5
fi
# The dealer's output is llama-dealer's gate; recorded here as provenance of the keys consumed.
md5sum server.dat client.dat > "$RESULTS/keys.md5"

echo "=== phase 2: server (2) + client (3), MEASURED ==="
T1=$(date +%s)
"$BIN" 2 127.0.0.1 > "$RESULTS/party2.log" 2>&1 &
P2=$!
echo "$P2" > "$RESULTS/party2.pid"

# The server binds and waits; the client connects. Starting the client first loses the race.
sleep 3

"$BIN" 3 127.0.0.1 > "$RESULTS/party3.log" 2>&1 &
P3=$!
echo "$P3" > "$RESULTS/party3.pid"

wait "$P2"; RC2=$?
wait "$P3"; RC3=$?
echo "=== party2 rc=$RC2  party3 rc=$RC3  online wall=$(( $(date +%s) - T1 ))s ==="
echo "$RC2" > "$RESULTS/exit_p2.txt"
echo "$RC3" > "$RESULTS/exit_p3.txt"

# The gate observable is the wire, not the output (T1s): the output is all zeros, so the
# comparison runs on protocol traffic through SocketBuf::read. Both parties emit their own line.
grep -h '^GATE|wire|' "$RESULTS/party2.log" "$RESULTS/party3.log" \
    | sort > "$RESULTS/gate.txt" 2>/dev/null
# Sorted so which party writes first cannot change the file; each line names its party.

# The value observable, kept beside the gate rather than as the gate.
grep -h '^GATE|logits|' "$RESULTS/party3.log" > "$RESULTS/gate_values.txt" 2>/dev/null || true

if [ ! -s "$RESULTS/gate.txt" ]; then
    echo "FAILED: no GATE|wire| line from either online party. Either this binary predates the" >&2
    echo "structural gate or the run did not reach finalize(). Check binary_md5.txt." >&2
    [ "$RC3" = 0 ] && RC3=5
fi
grep -h '^SEEDED|' "$RESULTS/party2.log" "$RESULTS/party3.log" > "$RESULTS/seeded.txt" 2>/dev/null
if [ ! -s "$RESULTS/seeded.txt" ]; then
    echo "FAILED: no SEEDED| marker; this run is not reproducible and cannot anchor a gate" >&2
    [ "$RC3" = 0 ] && RC3=5
fi

# 47 GB per run on a shared filesystem; removed to avoid the 141 GB / D-state stall three replicates caused.
rm -f server.dat client.dat

RC=$(( RC2 != 0 ? RC2 : RC3 ))
exit "$RC"
