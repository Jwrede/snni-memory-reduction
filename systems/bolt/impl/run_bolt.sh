#!/bin/bash
# BOLT, inside the container. Two parties on one node over loopback. ALICE (r=1) server/weights,
# BOB (r=2) client/one input; same binary, so this script writes party pids for the harness.
# Binding party expected ALICE but measured: the derive publishes whichever party had higher VmHWM.
set -uo pipefail

RESULTS="${BOLT_RESULTS:?need BOLT_RESULTS}"
BIN=/opt/EzPC/SCI/build/bin/BOLT_BERT
DATA="${BOLT_DATA:-/opt/bolt_data/quantize/sst-2}"
PORT="${BOLT_PORT:-8000}"

# The job-id-derived port is not guaranteed free, so a whole block (64 wide, above the largest
# thread constant) is checked via /proc/net/tcp and stepped; the binary binds a range, one per NL thread.
PORT_BLOCK=64
for _try in $(seq 0 20); do
    _base=$(( PORT + _try * PORT_BLOCK ))
    _busy=0
    for _o in $(seq 0 $(( PORT_BLOCK - 1 )) ); do
        _hex=$(printf '%04X' $(( _base + _o )) )
        if awk -v h=":$_hex" '$2 ~ h"$" || $3 ~ h"$" {found=1} END{exit !found}' /proc/net/tcp 2>/dev/null; then
            _busy=1; break
        fi
    done
    if [ "$_busy" = 0 ]; then
        [ "$_base" != "$PORT" ] && echo "port block at $PORT was not free; using $_base..$(( _base + PORT_BLOCK - 1 ))"
        PORT=$_base
        break
    fi
done
NUMCLASS="${BOLT_NUM_CLASS:-2}"
export SNNI_SEED="${SNNI_SEED:?need SNNI_SEED}"

mkdir -p "$RESULTS"
rm -f "$RESULTS"/party*.pid "$RESULTS/gate.txt"
md5sum "$BIN" | tee "$RESULTS/binary_md5.txt"

# The input sample is part of the measured program: pinned by default, overridable, recorded.
ID="${BOLT_ID:-}"
if [ -z "$ID" ]; then
    for f in "$DATA"/weights_txt/inputs_*_data.txt; do
        base=$(basename "$f"); id=${base#inputs_}; id=${id%_data.txt}
        if [ -f "$DATA/weights_txt/inputs_${id}_mask.txt" ]; then ID=$id; break; fi
    done
fi
[ -n "$ID" ] || { echo "FAILED: no input sample with both data and mask under $DATA" >&2; exit 2; }
echo "sample_id=$ID num_class=$NUMCLASS port=$PORT seed=$SNNI_SEED" | tee "$RESULTS/workload.txt"

export SNNI_GATE_FILE="$RESULTS/gate.txt"

echo "=== BOLT_BERT: ALICE (r=1, weights) then BOB (r=2, input)"
T0=$(date +%s)

# `env` execs the binary in its own process, so $! is the payload's pid and not a wrapper's.
env SNNI_SEED="$SNNI_SEED" SNNI_GATE_FILE="$SNNI_GATE_FILE" \
    "$BIN" r=1 p="$PORT" path="$DATA" num_class="$NUMCLASS" id="$ID" num_sample=1 \
    > "$RESULTS/party0_stdout.log" 2> "$RESULTS/party0_markers.txt" &
P0=$!
echo "$P0" > "$RESULTS/party0.pid"

# Wait for the listening socket by reading /proc/net/tcp for state 0A, without connecting: a
# probe connect is taken as the single peer connection and deadlocks the run.
PORT_HEX=$(printf '%04X' "$PORT")
LISTENING=0
for _ in $(seq 1 1200); do
    if awk -v p=":$PORT_HEX" '$2 ~ p"$" && $4 == "0A" {found=1} END{exit !found}' /proc/net/tcp 2>/dev/null; then
        LISTENING=1
        break
    fi
    # Checked inside the loop: a failed bind does not exit, so waiting the full timeout wastes minutes.
    if grep -qs "bind: Address already in use" \
            "$RESULTS/party0_stdout.log" "$RESULTS/party0_markers.txt" 2>/dev/null; then
        break
    fi
    kill -0 "$P0" 2>/dev/null || break
    sleep 0.5
done

# Distinguish the two exit reasons: a listener that never appeared vs a bind failure. The bind
# message lands in the marker file, not stdout, so both files are checked.
if grep -qs "bind: Address already in use" \
        "$RESULTS/party0_stdout.log" "$RESULTS/party0_markers.txt" 2>/dev/null; then
    echo "FAILED: party 0 could not bind port $PORT -- another process already holds it." >&2
    echo "        The port is derived as 8000 + SLURM_JOB_ID % 1000, so two jobs whose ids differ" >&2
    echo "        by a multiple of 1000 collide if they share a node. Re-run; the new job id gives" >&2
    echo "        a new port." >&2
    kill "$P0" 2>/dev/null
    exit 8
fi
if [ "$LISTENING" != 1 ]; then
    echo "FAILED: no listener on port $PORT ($PORT_HEX) after $(( SECONDS ))s." >&2
    echo "        Party 1 must not be started against a port this run does not own: it would" >&2
    echo "        connect to a stranger and both sides would wait out the time limit." >&2
    kill "$P0" 2>/dev/null
    exit 8
fi
echo "listener on port $PORT ($PORT_HEX) is up after $(( SECONDS ))s"

env SNNI_SEED="$SNNI_SEED" SNNI_GATE_FILE="$SNNI_GATE_FILE" \
    "$BIN" r=2 ip=127.0.0.1 p="$PORT" path="$DATA" num_class="$NUMCLASS" id="$ID" num_sample=1 \
    output="$RESULTS/pred.txt" \
    > "$RESULTS/party1_stdout.log" 2> "$RESULTS/party1_markers.txt" &
P1=$!
echo "$P1" > "$RESULTS/party1.pid"

# Watchdog: the 32 non-linear threads bind further ports outside the reserved block after startup,
# so a bind failure can arrive post-launch; end the run when the message appears rather than at the time limit.
_bolt_watch() {
    while kill -0 "$P0" 2>/dev/null || kill -0 "$P1" 2>/dev/null; do
        if grep -qs "bind: Address already in use" \
                "$RESULTS/party0_markers.txt" "$RESULTS/party1_markers.txt" \
                "$RESULTS/party0_stdout.log" "$RESULTS/party1_stdout.log" 2>/dev/null; then
            echo "FAILED: a bind failed after startup -- this build binds further ports for its" >&2
            echo "        non-linear threads, outside the block reserved for the base port. Both" >&2
            echo "        parties are now waiting for a peer that will not come; ending the run" >&2
            echo "        rather than waiting out the time limit." >&2
            kill "$P0" "$P1" 2>/dev/null
            return 0
        fi
        sleep 5
    done
}
_bolt_watch &
WATCH=$!

wait "$P0"; RC0=$?
wait "$P1"; RC1=$?
kill "$WATCH" 2>/dev/null; wait "$WATCH" 2>/dev/null

# Re-state the failure here: the watchdog kills, so a killed run must not look finished.
if grep -qs "bind: Address already in use" \
        "$RESULTS/party0_markers.txt" "$RESULTS/party1_markers.txt" \
        "$RESULTS/party0_stdout.log" "$RESULTS/party1_stdout.log" 2>/dev/null; then
    echo "FAILED: bind conflict during the run; this measurement is void" >&2
    exit 8
fi
echo "=== alice rc=$RC0  bob rc=$RC1  wall=$(( $(date +%s) - T0 ))s"
echo "$RC0" > "$RESULTS/exit_alice.txt"
echo "$RC1" > "$RESULTS/exit_bob.txt"

# --- the markers the harness requires, each checked separately -------------------------------
# Three seed markers, not one: PRG128 (share masks), PRG256 (KKOT OT), SEAL factory (keys/noise).
grep -h '^SEEDED|' "$RESULTS/party0_markers.txt" "$RESULTS/party1_markers.txt" \
    > "$RESULTS/seeded.txt" 2>/dev/null
for src in prg128 prg256 seal_prng; do
    grep -q "^SEEDED|$src|" "$RESULTS/seeded.txt" 2>/dev/null || {
        echo "FAILED: no SEEDED|$src| marker; that draw source is unpinned in this binary" >&2
        [ "$RC0" = 0 ] && RC0=5
    }
done

grep -h '^WORKLOAD|' "$RESULTS/party0_markers.txt" "$RESULTS/party1_markers.txt" \
    >> "$RESULTS/workload.txt" 2>/dev/null

# --- the gate -------------------------------------------------------------------------------
# BOB writes the reconstructed class scores at full precision to SNNI_GATE_FILE; a thin two-value
# observable, so the transcript volume below carries the weight.
NG=$(wc -l < "$RESULTS/gate.txt" 2>/dev/null || echo 0)
echo "gate values: $NG (expected $NUMCLASS)"
if [ "$NG" != "$NUMCLASS" ]; then
    echo "FAILED: gate observable missing or short ($NG of $NUMCLASS)" >&2
    [ "$RC0" = 0 ] && RC0=5
fi
if grep -q '^GATEFAIL|' "$RESULTS/party1_markers.txt" 2>/dev/null; then
    echo "FAILED: BOB could not write its gate file" >&2
    [ "$RC0" = 0 ] && RC0=5
fi
# A gate of all zeros cannot fail (SHARK published one before this check existed).
if [ "$NG" -gt 0 ] && ! awk 'BEGIN{nz=0} {if ($1+0 != 0) nz++} END{exit !(nz>0)}' "$RESULTS/gate.txt"; then
    echo "FAILED: every gate value is zero" >&2
    [ "$RC0" = 0 ] && RC0=6
fi

# --- the transcript invariant, beside the gate and not part of it ----------------------------
# The protocol's own per-phase byte/round counts: exact and data-independent where the numeric
# observable has a tolerance, so the two check semantics through different mechanisms.
grep -hE '^> \[NETWORK\]' "$RESULTS/party0_stdout.log" "$RESULTS/party1_stdout.log" \
    > "$RESULTS/comm.txt" 2>/dev/null
if [ ! -s "$RESULTS/comm.txt" ]; then
    echo "WARNING: no [NETWORK] lines; the transcript invariant is missing from this run" >&2
fi
wc -l < "$RESULTS/comm.txt" 2>/dev/null

RC=$(( RC0 != 0 ? RC0 : RC1 ))
exit "$RC"
