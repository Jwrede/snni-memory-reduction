#!/bin/bash
# SHARK in the container; only phase 2 (online role) is measured.
#   phase 1: dealer (party 2) writes server.dat/client.dat
#   phase 2: server (0) and client (1) run concurrently, each reading its file
# Split forced by the protocol (init::gen vs init::eval). party0/party1.pid written only in phase 2.
set -uo pipefail

RESULTS="${SHARK_RESULTS:?need SHARK_RESULTS}"
WORK="${SHARK_WORK:?need SHARK_WORK}"
BIN=/root/shark/build/benchmark-bert_instrumented

export SHARK_SEQ_LEN="${SHARK_SEQ_LEN:-128}"
export SHARK_N_LAYERS="${SHARK_N_LAYERS:-12}"

cd "$WORK" || exit 9
mkdir -p "$RESULTS"
rm -f server.dat client.dat "$RESULTS"/party0.pid "$RESULTS"/party1.pid

md5sum "$BIN" | tee "$RESULTS/binary_md5.txt"
cat /root/shark_commit.txt > "$RESULTS/upstream_commit.txt" 2>/dev/null

echo "=== phase 1: dealer (party 2), NOT measured ==="
T0=$(date +%s)
"$BIN" 2 > "$RESULTS/dealer_stdout.log" 2> "$RESULTS/dealer_markers.txt"
RCD=$?
echo "dealer rc=$RCD  wall=$(( $(date +%s) - T0 ))s"
echo $(( $(date +%s) - T0 )) > "$RESULTS/dealer_wall_seconds.txt"

# SIZES of the key files recorded (free); not md5'd (~100 GB, minutes/replicate, and the online
# output is the gate). A truncated key file gave a starved replicate with a normal peak elsewhere.
stat -c '%n %s' server.dat client.dat > "$RESULTS/keyfiles.txt" 2>/dev/null
cat "$RESULTS/keyfiles.txt"
if [ "$RCD" != 0 ] || [ ! -s server.dat ] || [ ! -s client.dat ]; then
    echo "FAILED: key generation did not produce both preprocessing files; nothing to measure" >&2
    exit 5
fi

echo "=== phase 2: server (0) + client (1), MEASURED ==="
T1=$(date +%s)

# PARTY 0 FIRST: init::eval sends the server into waitForPeer and the client into connect, so the
# client cannot start first. Party 0 is the freeze leader (atomic capture) and had the larger peak
# (54.3 vs 53.4 GB), so its decomposition selects the lever.
SHARK_OUT_DUMP="$RESULTS/output_party0.txt" \
    "$BIN" 0 > "$RESULTS/party0_stdout.log" 2> "$RESULTS/party0_markers.txt" &
P0=$!
echo "$P0" > "$RESULTS/party0.pid"

sleep 3

SHARK_OUT_DUMP="$RESULTS/output_party1.txt" \
    "$BIN" 1 > "$RESULTS/party1_stdout.log" 2> "$RESULTS/party1_markers.txt" &
P1=$!
echo "$P1" > "$RESULTS/party1.pid"

wait "$P0"; RC0=$?
wait "$P1"; RC1=$?
echo "=== party0 rc=$RC0  party1 rc=$RC1  online wall=$(( $(date +%s) - T1 ))s ==="
echo $(( $(date +%s) - T1 )) > "$RESULTS/online_wall_seconds.txt"
echo "$RC0" > "$RESULTS/exit_p0.txt"
echo "$RC1" > "$RESULTS/exit_p1.txt"

# THE GATE OBSERVABLE. Both online parties reconstruct the same output, so both emit into gate.txt
# (a per-party divergence stays visible). Full vectors kept beside it as evidence.
grep -h '^GATE|' "$RESULTS/party0_markers.txt" "$RESULTS/party1_markers.txt" \
    > "$RESULTS/gate.txt" 2>/dev/null
if [ ! -s "$RESULTS/gate.txt" ]; then
    echo "FAILED: no GATE| line from either online party. Either this binary predates the gate" >&2
    echo "patch or the run did not reach its output. Check binary_md5.txt." >&2
    [ "$RC0" = 0 ] && RC0=5
fi
for f in "$RESULTS/output_party0.txt" "$RESULTS/output_party1.txt"; do
    if [ ! -s "$f" ]; then
        echo "FAILED: missing full output dump $f" >&2
        [ "$RC0" = 0 ] && RC0=5
    fi
done

# A CONSTANT OUTPUT IS A GATE THAT CANNOT FAIL. This system's first determinism check reproduced
# byte-perfectly with 98,304 zeros (input/weights unwritten before the mask), the same defect found
# on SIGMA-GPU. The binary reports how many outputs differ from the first; a constant run is refused.
grep -h '^GATEQ|' "$RESULTS/party0_markers.txt" "$RESULTS/party1_markers.txt" \
    > "$RESULTS/gate_quality.txt" 2>/dev/null
if [ ! -s "$RESULTS/gate_quality.txt" ]; then
    echo "FAILED: no GATEQ| line; cannot tell whether the gate observable is degenerate" >&2
    [ "$RC0" = 0 ] && RC0=5
else
    cat "$RESULTS/gate_quality.txt"
    if grep -q 'differing_from_first=0$' "$RESULTS/gate_quality.txt"; then
        echo "FAILED: the reconstructed output is CONSTANT. That is a gate that cannot fail." >&2
        [ "$RC0" = 0 ] && RC0=6
    fi
fi

grep -h '^SEEDED|' "$RESULTS/party0_markers.txt" "$RESULTS/party1_markers.txt" \
    > "$RESULTS/seeded.txt" 2>/dev/null
if [ ! -s "$RESULTS/seeded.txt" ]; then
    echo "FAILED: no SEEDED| marker; this run is not reproducible and cannot anchor a gate" >&2
    [ "$RC0" = 0 ] && RC0=5
fi

# A STEP THAT IS NOT THE BASELINE MUST ANNOUNCE ITS LEVER. Levers are compile-time source changes in
# their own image, but a wrong image gives a plausible number under the wrong name. Each lever prints
# a `LEVER|` line, and a non-baseline step whose run carries none is refused here.
if [ -n "${SHARK_STEP:-}" ] && [ "${SHARK_STEP%%_*}" != "s0" ]; then
    grep -h '^LEVER|' "$RESULTS/party0_markers.txt" "$RESULTS/party1_markers.txt" \
        > "$RESULTS/lever.txt" 2>/dev/null
    if [ ! -s "$RESULTS/lever.txt" ]; then
        echo "FAILED: step $SHARK_STEP produced no LEVER| marker, so this run measures some other" >&2
        echo "binary under its name. Check binary_md5.txt against the parent step's." >&2
        [ "$RC0" = 0 ] && RC0=7
    else
        cat "$RESULTS/lever.txt"
    fi
fi

# The batch integrity check is SHARK's own cryptographic invariant over every authenticated share
# (common.cpp always_assert), reached from output::eval; exit 0 means it passed. A second,
# independent statement the output hash does not make.
{ echo "party0_rc=$RC0"; echo "party1_rc=$RC1"; } > "$RESULTS/batch_check.txt"

# ~100 GB of preprocessing per run on a shared network filesystem. Left behind, three replicates
# would be 300 GB, and a /scratch stall is what produced unkillable D-state processes here once.
rm -f server.dat client.dat

RC=$(( RC0 != 0 ? RC0 : RC1 ))
exit "$RC"
