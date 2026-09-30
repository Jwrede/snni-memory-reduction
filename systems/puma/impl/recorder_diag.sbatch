#!/bin/bash
#SBATCH --job-name=pumadiag
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/puma/recdiag-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/puma/recdiag-%j.out
#
#   sbatch --export=ALL,STEP=s0_default,SNNI_PM_MIN=65536,SNNI_PM_FULLSCAN=1 recorder_diag.sbatch
#
# Diagnosis of a missed 112 MiB numpy array (2026-08-02): ctypes-direct malloc resolves inside libc
# and bypasses the PLT interposer (probe property). Allocates the same bytes via ctypes, numpy and
# bytearray and prints the recorder counters after each.
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/puma
AR=$C/attrib_resident
SIF=$B/sif/puma-$STEP.sif
# One directory per variant, run chained not parallel: sharing one directory made variants overwrite each other's tables.
OUT=$B/recdiag/$STEP/${SNNI_DIAG_WHICH:-all}

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
[ -r "$SIF" ] || { echo "FAILED: no image at $SIF"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT" || exit 1

cat > "$OUT/probe.py" <<'EOF'
import ctypes, os, sys, time
import numpy as np

TABLE = os.environ["SNNI_PM_TABLE"].replace("%d", str(os.getpid()))

def stats(tag):
    # pmrec writes its counters into the table header; read them without pmsample so this stays a
    # statement about the RECORDER rather than about the sampler.
    try:
        with open(TABLE, "rb") as fh:
            head = fh.read(4096)
        txt = "".join(chr(b) if 32 <= b < 127 else " " for b in head)
    except Exception as e:
        txt = "unreadable: %s" % e
    sys.stderr.write("STAGE|%s|table_head=%s\n" % (tag, txt[:200].strip()))
    sys.stderr.flush()

N = 64 << 20
stats("start")

WHICH = os.environ.get("SNNI_DIAG_WHICH", "all")

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.malloc.restype = ctypes.c_void_p
p = None
a = None
b = None
if WHICH in ("all", "libc"):
    p = libc.malloc(ctypes.c_size_t(N))
    ctypes.memset(ctypes.c_void_p(p), 1, N)
    sys.stderr.write("STAGE|libc_malloc|ptr=%x\n" % p)
if WHICH in ("all", "numpy"):
    a = np.ones(N, dtype=np.uint8)      # numpy's own data allocator
    a[::4096] = 2
    sys.stderr.write("STAGE|numpy_ones|nbytes=%d\n" % a.nbytes)
if WHICH in ("all", "bytearray"):
    b = bytearray(N)                    # CPython object allocator for a large block
    b[::4096] = b"\x02" * (len(b) // 4096)
    sys.stderr.write("STAGE|bytearray|len=%d\n" % len(b))
sys.stderr.write("STAGE|which=%s\n" % WHICH)

sys.stderr.write("PROBE_READY pid=%d\n" % os.getpid())
sys.stderr.flush()
time.sleep(25)
print(WHICH, a is not None, b is not None, p is not None)
EOF

apptainer exec --bind "$AR":/ar --bind "$OUT":"$OUT" "$SIF" bash -c \
    "export LD_PRELOAD=/ar/pmrec.so SNNI_PM_TABLE=$OUT/pmtable_%d SNNI_PM_MIN=${SNNI_PM_MIN:?need floor}; exec python3 $OUT/probe.py" \
    > "$OUT/probe.out" 2> "$OUT/probe.err" &
APP=$!

PID=""
for _ in $(seq 1 600); do
    PID=$(pgrep -u "$(id -u)" -f "$OUT/probe.py" | head -1)
    [ -n "$PID" ] && grep -q PROBE_READY "$OUT/probe.err" 2>/dev/null && break
    sleep 0.1
done
[ -n "$PID" ] || { echo "FAILED: probe never appeared"; cat "$OUT/probe.err"; exit 4; }

echo "=== the recorder's own view, stage by stage"
grep '^STAGE|' "$OUT/probe.err"

echo
echo "=== pmsample against the live probe"
export SNNI_PM_FULLSCAN
"$AR/pmsample" "$PID" "$OUT/pmtable_$PID" "$OUT/pm.maps" > "$OUT/resident.txt" 2>"$OUT/pmsample.err"
echo "pmsample rc=$?"
grep '^# live_allocations' "$OUT/resident.txt"
echo "--- the five largest recorded sites (resident allocated count site)"
grep -v '^#' "$OUT/resident.txt" | sort -rn | head -5

echo
echo "=== where the process's resident memory actually is, per mapping"
awk '{print}' "/proc/$PID/status" 2>/dev/null | grep -E 'VmRSS|VmHWM'
grep -E 'Rss:' "/proc/$PID/smaps" 2>/dev/null | awk '{s+=$2} END{printf "smaps total Rss = %d kB\n", s}'

echo
echo "=== does python3 itself resolve malloc through the interposer"
apptainer exec --bind "$AR":/ar "$SIF" bash -c \
    'LD_PRELOAD=/ar/pmrec.so LD_DEBUG=bindings LD_DEBUG_OUTPUT=/tmp/lddbg python3 -c "pass" 2>/dev/null;
     grep -h "binding file .*python3.* to /ar/pmrec.so: normal symbol .malloc" /tmp/lddbg.* 2>/dev/null | head -3;
     echo "(no line above means python3 does not bind malloc to the interposer)"'

wait "$APP" 2>/dev/null
echo "DIAG_DONE"
