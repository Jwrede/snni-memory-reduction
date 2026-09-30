#!/bin/bash
#SBATCH --job-name=llamad
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/llama-dealer/slurm-%x-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/llama-dealer/slurm-%x-%j.err
#
# LLAMA dealer on PALMA (same procedure and images as the server C harness).
#
#   sbatch --export=ALL,STEP=d0_default llama_dealer.sbatch            # clean run
#   sbatch --export=ALL,STEP=d0_default,PMREC=1 llama_dealer.sbatch    # resident attribution
#
# Partition express: same node type as normal (2 x 18-core Skylake, RealMemory=95000).
# Differences from the podman harness: --exclusive; OMP_NUM_THREADS=16 (as server C, not 36);
# poller outside the container (apptainer shares the host PID namespace).
set -uo pipefail

BASE=/scratch/tmp/j_wred02/snni_campaign/llama-dealer
AR=/scratch/tmp/j_wred02/snni_campaign/attrib_resident
CAMP=/scratch/tmp/j_wred02/snni_campaign

# Checked before the run, not discovered after: a missing poller script previously produced a
# silent no-op (payload completed, no trace existed).
for f in "$CAMP/poll_peak.sh"; do
    [ -f "$f" ] || { echo "FAILED: instrument missing: $f" >&2; exit 4; }
done
if [ -n "${PMREC:-}" ]; then
    for f in "$AR/pmrec.so" "$AR/pmsample"; do
        [ -f "$f" ] || { echo "FAILED: PMREC requested but $f is missing" >&2; exit 4; }
    done
fi
STEP="${STEP:?need STEP}"
IMG_STEP="${STEP%_pmrec}"
SIF="${SIF:-$BASE/sif/llama-dealer-$IMG_STEP.sif}"
# RUN=<n> separates replicates into their own directories, so the published median (of three
# measured replicates) and the spread are computable rather than collapsed to the worst trace.
RES="$BASE/results/$STEP${RUN:+/run$RUN}"
WORK="${WORK:-/scratch/tmp/j_wred02/llama_dealer_work/$STEP${RUN:+_run$RUN}}"
BIN=/opt/EzPC/GPU-MPC/ext/sytorch/build/bertbenchmark

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

[ -f "$SIF" ] || { echo "ERROR: $SIF missing; run build_sifs.sh" >&2; exit 2; }
# A re-run's previous traces are moved aside, not deleted (evidence of reproducibility) and not
# left in place, since the peak is taken as the max over every trace in the directory.
if [ -d "$RES" ] && [ -n "$(ls -A "$RES" 2>/dev/null)" ]; then
    PREV="$BASE/results/.superseded/$STEP.$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$(dirname "$PREV")"
    mv "$RES" "$PREV"
    echo "previous run moved aside to $PREV"
fi
mkdir -p "$RES" "$WORK"
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

{
  echo "step=$STEP"
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "role=dealer (party 1), measured alone"
  echo "machine=palma:${SLURM_JOB_PARTITION:-normal} node=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-na}"
  echo "image=$SIF"
  echo "image_digest=$(sha256sum "$SIF" | awk '{print $1}')"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  echo "malloc_arena_max=$MALLOC_ARENA_MAX"
  echo "malloc_mmap_threshold=${MALLOC_MMAP_THRESHOLD_:-unset(dynamic)}"
  echo "malloc_trim_threshold=${MALLOC_TRIM_THRESHOLD_:-unset(dynamic)}"
  echo "vram=NA (CPU system, single pool -- the degenerate case of the two-pool protocol)"
  echo "model=bert-base n_embd=768 n_layer=12 bw=51 n_seq=128 scale=12"
  echo "attrib_host=$([ -n "${PMREC:-}" ] && echo pmrec_pagemap || echo off)"
  echo "binary_md5=$(apptainer exec "$SIF" md5sum "$BIN" | awk '{print $1}')"
} > "$RES/RUN_META"
cat "$RES/RUN_META"

# The image is the archive of the source state, so the diff is taken from it, not from a checkout.
apptainer exec "$SIF" bash -c 'cd /opt/EzPC && git status --short' > "$RES/src.status" 2>/dev/null
apptainer exec "$SIF" bash -c 'cd /opt/EzPC && git diff'          > "$RES/src.diff"   2>/dev/null
echo "src_diff_md5=$(md5sum "$RES/src.diff" | awk '{print $1}')" >> "$RES/RUN_META"

cd "$WORK" || exit 2
rm -f server.dat client.dat
echo "=== dealer run (party 1)  pmrec=${PMREC:-off} ==="
T0=$(date +%s)

# Environment set by an inner shell that execs the binary, not apptainer --env (which manages
# LD_PRELOAD itself in some configs and can silently drop it); exec keeps the pid stable for the poller.
INNER="export OMP_NUM_THREADS=$OMP_NUM_THREADS MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"
# Allocator levers passed only when explicitly set (empty is not unset for glibc). ALT_PRELOAD
# swaps the allocator for diagnostic off-path runs, and is incompatible with PMREC (both are LD_PRELOAD).
if [ -n "${ALT_PRELOAD:-}" ]; then
    if [ -n "${PMREC:-}" ]; then
        echo "FAILED: ALT_PRELOAD and PMREC cannot both be set" >&2
        exit 6
    fi
    INNER="$INNER; export LD_PRELOAD=$ALT_PRELOAD"
fi
if [ -n "${MALLOC_MMAP_THRESHOLD_:-}" ]; then
    INNER="$INNER MALLOC_MMAP_THRESHOLD_=$MALLOC_MMAP_THRESHOLD_"
fi
if [ -n "${MALLOC_TRIM_THRESHOLD_:-}" ]; then
    INNER="$INNER MALLOC_TRIM_THRESHOLD_=$MALLOC_TRIM_THRESHOLD_"
fi
if [ -n "${PMREC:-}" ]; then
    INNER="$INNER; export LD_PRELOAD=/ar/pmrec.so SNNI_PM_TABLE=$RES/pmtable"
    # SNNI_PM_MIN required from campaign.env, never defaulted here (a stale default is silent, a
    # missing one is loud); the old 1 MiB default misranked objects at d2 (measured).
    INNER="$INNER SNNI_PM_MIN=${SNNI_PM_MIN:?campaign.env must supply the recorder floor}"
fi
INNER="$INNER; exec $BIN 1"

apptainer exec \
    --pwd "$WORK" \
    --bind "$WORK":"$WORK" --bind "$RES":"$RES" \
    ${PMREC:+--bind "$AR":/ar} \
    "$SIF" bash -c "$INNER" > "$RES/stdout.log" 2> "$RES/stderr.log" &
APP_PID=$!

# Attaches to the process that holds the memory, confirmed as a descendant of this job's
# apptainer process rather than matched by name alone (cost four runs elsewhere when done wrong).
descends_from() {
    local pid="$1" want="$2" ppid n=0
    while [ "$pid" -gt 1 ] && [ "$n" -lt 40 ]; do
        [ "$pid" = "$want" ] && return 0
        ppid=$(awk '/^PPid:/{print $2}' "/proc/$pid/status" 2>/dev/null) || return 1
        [ -n "$ppid" ] || return 1
        pid="$ppid"; n=$((n + 1))
    done
    return 1
}

# Matched on /proc/<pid>/exe, not cmdline: both the apptainer process and inner shell carry
# "bertbenchmark" in their cmdline, so only exe identifies the actual process.
PAYLOAD=""
for _ in $(seq 1 300); do
    for c in $(pgrep -u "$(id -u)" -f bertbenchmark 2>/dev/null); do
        [ "$c" = "$APP_PID" ] && continue
        # Plain readlink, not -f: canonicalising the container-internal path fails on the host and
        # returns empty, filtering out the real process. "(deleted)" suffix is tolerated.
        case "$(readlink "/proc/$c/exe" 2>/dev/null)" in
            */bertbenchmark|*"/bertbenchmark (deleted)") ;;
            *) continue ;;
        esac
        if descends_from "$c" "$APP_PID"; then PAYLOAD="$c"; break; fi
    done
    [ -n "$PAYLOAD" ] && break
    kill -0 "$APP_PID" 2>/dev/null || break
    sleep 0.1
done

POLL_PID=""
if [ -n "$PAYLOAD" ]; then
    echo "poller attached to $PAYLOAD (apptainer pid $APP_PID)" | tee -a "$RES/poller_attach.log"
    # SNNI_PM_FULLSCAN makes pmsample scan every mapping so unnamed memory is located rather than
    # left as "anon minus named heap"; the mode is stamped into resident.txt.
    SNNI_PM_TABLE="${PMREC:+$RES/pmtable}" SNNI_PM_SAMPLE="${PMREC:+$AR/pmsample}" \
    SNNI_PM_FULLSCAN="${SNNI_PM_FULLSCAN:?campaign.env must supply the fullscan switch}" \
        bash "$CAMP/poll_peak.sh" "$PAYLOAD" "${POLL_MS:?campaign.env must supply the poll interval}" "$RES/poll_dealer" > "$RES/poller.log" 2>&1 &
    POLL_PID=$!
else
    echo "ERROR: no bertbenchmark process under $APP_PID; this run has no peak" >&2
fi

RC=0; wait "$APP_PID" || RC=$?
[ -n "$POLL_PID" ] && wait "$POLL_PID" 2>/dev/null
T1=$(date +%s)

# Keys must match the baseline (a lever changes only when memory frees, never what's generated).
# md5sum first, then stat: reading the whole file forces network-filesystem writeback to settle,
# since stat alone once reported a wrong in-progress size.
md5sum server.dat client.dat > "$RES/keys.md5" 2>/dev/null || true
SZS=$(stat -c %s server.dat 2>/dev/null || echo 0)
SZC=$(stat -c %s client.dat 2>/dev/null || echo 0)
rm -f server.dat client.dat

echo $(( T1 - T0 )) > "$RES/wall_seconds.txt"
PEAK_RSS=$(awk 'NR>1 && $2 ~ /^[0-9]+$/{if($2>m)m=$2}END{print m+0}' "$RES/poll_dealer/vmrss.log")
{
  echo "step=$STEP  exit=$RC  wall=$(( T1 - T0 ))s"
  echo "peak_host_kb=$PEAK_RSS"
  echo "peak_vram_kb=NA"
  echo "server_dat_B=$SZS  client_dat_B=$SZC"
} > "$RES/PEAK.txt"
cat "$RES/PEAK.txt"
cat "$RES/keys.md5" 2>/dev/null || echo "no keyfiles (dealer failed?)"
rm -f "$RES/pmtable"

# A run with no peak is not successful regardless of exit code: the first PALMA attempt at d0
# exited 0 with correct keys but no measurement, since the poller never attached.
if [ -z "$PAYLOAD" ]; then
    echo "FAILED: the poller never attached, so this run has no peak and is not a measurement" >&2
    exit 3
fi
if [ ! -s "$RES/poll_dealer/vmrss.log" ]; then
    echo "FAILED: no poll trace at $RES/poll_dealer/vmrss.log" >&2
    exit 3
fi
if [ -z "$PEAK_RSS" ] || [ "$PEAK_RSS" -le 0 ]; then
    echo "FAILED: poll trace exists but yields no peak (PEAK_RSS='$PEAK_RSS')" >&2
    exit 3
fi
# With PMREC on, the object table is half the measurement; without it the step could be published
# with a peak and no decomposition, which the procedure does not allow.
if [ -n "${PMREC:-}" ] && [ ! -s "$RES/poll_dealer/peaks/resident.txt" ]; then
    echo "FAILED: PMREC was requested but no resident table was written" >&2
    exit 3
fi
# Gate observable must exist: one starved concurrent run wrote zero-byte keys and exited 0 with
# a plausible peak, which without this check would have been published as a valid replicate.
if [ ! -s "$RES/keys.md5" ]; then
    echo "FAILED: no keys.md5; the dealer produced no gate observable" >&2
    exit 5
fi
if [ "$SZS" -ne 23003004064 ] || [ "$SZC" -ne 23909760144 ]; then
    echo "FAILED: key files are the wrong size (server=$SZS client=$SZC); expected" \
         "23003004064 and 23909760144" >&2
    exit 5
fi
exit "$RC"
