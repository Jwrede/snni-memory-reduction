#!/bin/bash
# SIGMA two-party run inside the container; both parties share ONE device and talk over loopback.
# Adapted from run_sigma_shared.sh; the addition is that IT WRITES ITS PARTY PIDS, because both
# parties are the same binary/name (no cmdline match) and the poller must attach to the memory
# holders. Apptainer shares the host PID namespace, so these pids are what the host poller attaches to.
set -uo pipefail

MODEL="${SIGMA_MODEL:-bert-base}"
SEQ="${SIGMA_SEQ:-128}"
THREADS="${SIGMA_THREADS:-16}"
PEER="127.0.0.1"
RESULTS="${SIGMA_RESULTS:-/results}"

cd /sigma/EzPC/GPU-MPC/experiments/sigma || exit 9
mkdir -p output/P0/models output/P1/models "$RESULTS" "$RESULTS/dev_party0" "$RESULTS/dev_party1"

echo "=== SIGMA two-party shared-device run: model=$MODEL seq=$SEQ threads=$THREADS ==="
# Record which binary actually runs: the image bakes one in and a measured step bind-mounts its own
# over it; running the baked-in one would publish the baseline under the step's name.
md5sum ./sigma | tee "$RESULTS/binary_md5.txt"
nvidia-smi -L 2>/dev/null || true

# WAIT FOR PORT 42003 TO BE FREE BEFORE LAUNCHING. The port is compiled in (sigma_comms.cpp:157), so
# SLURM-chained replicates can hit a socket still in TIME_WAIT: party 0 fails to bind, party 1 waits
# forever, the job runs to its limit (measured job 45862838). Reads /proc/net/tcp AND tcp6 (a
# v4-mapped AF_INET6 listener shows only in tcp6), reading rather than connecting so a mid-handshake
# peer is undisturbed.
PORT_HEX=$(printf '%04X' 42003)
_port_busy() {
    awk -v h=":$PORT_HEX" '$2 ~ h"$" || $3 ~ h"$" {found=1} END{exit !found}' \
        /proc/net/tcp /proc/net/tcp6 2>/dev/null
}
for _ in $(seq 1 120); do
    _port_busy || break
    sleep 1
done
if _port_busy; then
    echo "FAILED: port 42003 is still in use after 120 s. This program cannot be told to use" >&2
    echo "        another one, so the run is refused rather than started into a bind failure" >&2
    echo "        that would only surface as a time limit." >&2
    exit 8
fi

# ONE DEVICE-RECORDER OUTPUT PER PARTY: same binary on the same card, so a shared dir would have the
# last finisher overwrite the other and the decomposition would describe a different process than the
# peak. The derive picks the leader's directory by name.
CUDA_VISIBLE_DEVICES=0 SNNI_CUDAREC_OUT="$RESULTS/dev_party0" ./sigma "$MODEL" "$SEQ" 0 "$PEER" "$THREADS" \
    > "$RESULTS/party0.log" 2>&1 &
P0=$!
echo "$P0" > "$RESULTS/party0.pid"

# The peer needs a moment to bind before the other connects. RAISED FROM 3 s TO 10 s: on the full
# H200 one of the first two runs died in recvBytes on numRead==toRead while the other completed (a
# setup race), and both parties allocate a 40 GiB device pool before they talk.
sleep 10

# Check the bind AFTER THE FACT: party 0 does not exit on bind failure (it prints and keeps running),
# so nothing downstream would notice until the time limit. Refusing here costs seconds.
if grep -qs "bind: Address already in use" "$RESULTS/party0.log"; then
    echo "FAILED: party 0 could not bind port 42003 even though it was free before launch." >&2
    echo "        Party 1 is not started: it would connect to a stranger and both sides would" >&2
    echo "        wait out the limit." >&2
    kill "$P0" 2>/dev/null
    exit 8
fi

CUDA_VISIBLE_DEVICES=0 SNNI_CUDAREC_OUT="$RESULTS/dev_party1" ./sigma "$MODEL" "$SEQ" 1 "$PEER" "$THREADS" \
    > "$RESULTS/party1.log" 2>&1 &
P1=$!
echo "$P1" > "$RESULTS/party1.pid"

wait "$P0"; RC0=$?
wait "$P1"; RC1=$?
echo "=== P0 exit=$RC0  P1 exit=$RC1 ==="
echo "$RC0" > "$RESULTS/exit_p0.txt"
echo "$RC1" > "$RESULTS/exit_p1.txt"

# Collect the protocol's reports explicitly: they are written only inside the container, and Total
# Comm is printed through a rounding helper, so no grep on the log matches it exactly.
cp -r output/P0 "$RESULTS/P0" 2>/dev/null
cp -r output/P1 "$RESULTS/P1" 2>/dev/null
find output \( -name dealer.txt -o -name evaluator.txt \) 2>/dev/null | while read -r f; do
    d="$RESULTS/$(dirname "$f")"; mkdir -p "$d"; cp "$f" "$d/"
done

# THE GATE ON THIS SYSTEM IS STRUCTURAL (tier T1s; derivation in MANIFEST.md). The value observable
# does not work, measured not assumed: with the seed pinned and input/weights filled (SEEDED|, FILL|)
# the output is still 98,304 exact zeros, so runs agree byte-for-byte carrying no information
# (off-path-runs/p_det/; same shape and cause as LLAMA online). The transcript is assembled by the
# SBATCH (invariants.txt is written after this exits); this block writes only the EVIDENCE.
#
grep -h '^GATE|logits|' "$RESULTS/party0.log" > "$RESULTS/gate_values.txt" 2>/dev/null
grep -h '^GATE|logits|' "$RESULTS/party1.log" > "$RESULTS/gate_party1.txt" 2>/dev/null
grep -h '^GATE_NONZERO|' "$RESULTS/party0.log" > "$RESULTS/gate_nonzero.txt" 2>/dev/null

# The seed marker exists so that a seed which failed to take cannot look like one that worked.
grep -h '^SEEDED|' "$RESULTS/party0.log" > "$RESULTS/seeded.txt" 2>/dev/null
if [ ! -s "$RESULTS/seeded.txt" ]; then
    echo "FAILED: no SEEDED| marker; this run is not reproducible and cannot anchor a gate" >&2
    [ "$RC0" = 0 ] && RC0=5
fi

RC=$(( RC0 != 0 ? RC0 : RC1 ))
exit "$RC"
