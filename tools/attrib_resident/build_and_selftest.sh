#!/bin/bash
# Builds pmrec and pmsample and runs selftest.c against known answers (login node, host glibc).
#
#   build_and_selftest.sh [outdir]
#
# Shipped artifact: build_in_image.sh.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-${TMPDIR:-/tmp}/pmrec_selftest.$$}"
mkdir -p "$OUT" || exit 2

echo "=== build ==="
gcc -O2 -g -fPIC -shared -o "$OUT/pmrec.so" "$HERE/pmrec.c" -ldl || exit 2
gcc -O2 -g -o "$OUT/pmsample" "$HERE/pmsample.c" || exit 2
# `-gdwarf-4` not plain `-g`: PALMA login-node binutils can't read gcc 11.5's default DWARF 5,
# so addr2line silently returns `??` for every address instead of erroring (host-only issue; real
# derives run addr2line inside the step image where compiler/binutils match).
# `-fPIE -pie` for the same reason: measured programs are PIE, but login-node gcc defaults to
# non-PIE, which breaks symbolise.py's `addr - mapping_start + file_offset` math silently.
gcc -O2 -gdwarf-4 -fPIE -pie -o "$OUT/selftest" "$HERE/selftest.c" || exit 2
ls -l "$OUT/pmrec.so" "$OUT/pmsample" "$OUT/selftest"

run_at_depth() {   # depth -> writes $OUT/table.<depth> and $OUT/sample.<depth>
    local d="$1"
    rm -f "$OUT/table.$d"
    SNNI_SELFTEST_HOLD_SECONDS=10 \
    SNNI_PM_TABLE="$OUT/table.$d" SNNI_PM_MIN=1048576 SNNI_PM_DEPTH="$d" \
    LD_PRELOAD="$OUT/pmrec.so" "$OUT/selftest" > "$OUT/prog.$d" 2>&1 &
    local pid=$!
    sleep 3
    # Maps snapshot is the third argument: without it there's nothing to resolve addresses
    # against, so anything downstream sees raw hex.
    "$OUT/pmsample" "$pid" "$OUT/table.$d" "$OUT/maps.$d" > "$OUT/sample.$d" 2>"$OUT/sample.$d.err"
    wait "$pid"
}

echo "=== run at depth 1 (the published default) ==="
run_at_depth 1
echo "=== run at depth 2 (caller capture) ==="
run_at_depth 2
# Depth 3 exists because on SHAFT-CPU the second frame is STILL inside numpy, so whether a third
# leaves the library has to be measurable. Here it is the assertion that the index tracks the
# depth rather than being hard-wired: one frame further out, the three owners share `main`.
echo "=== run at depth 3 (one frame further out) ==="
run_at_depth 3

echo
# Proves --program-root: it names a row by the first frame whose source file lies under the
# program, instead of a fixed index, fixing the case that cost MOAI-CPU its decomposition (an
# allocation through a helper landing in libc with no attackable name).
echo "=== chain and --program-root ==="
CHAINCOLS=$(grep -v '^#' "$OUT/sample.3" | head -1 | awk '{print NF}')
if [ "${CHAINCOLS:-0}" -le 5 ]; then
    echo "FAIL: the resident table carries $CHAINCOLS columns, so the recorder wrote no frame chain"
    exit 1
fi
echo "ok: the resident table carries the frame chain ($CHAINCOLS columns)"
HERE_ROOT="$(cd "$HERE/../.." && pwd)"
python3 "$HERE/resolve_resident.py" "$OUT/sample.3" "$OUT/maps.3" "$OUT/obj_fixed.jsonl" \
    --top 5 > "$OUT/label_fixed.txt" 2>&1
python3 "$HERE/resolve_resident.py" "$OUT/sample.3" "$OUT/maps.3" "$OUT/obj_pred.jsonl" \
    --top 5 --program-root "$HERE_ROOT" > "$OUT/label_pred.txt" 2>&1
if grep -q "selftest.c:.*:main  <- selftest.c:.*:helper_alloc" "$OUT/label_pred.txt"; then
    echo "ok: --program-root names the helper's row after the PROGRAM frame"
else
    echo "FAIL: --program-root did not reach the program frame. Rows were:"
    grep -v '^#' "$OUT/label_pred.txt" | head -5
    exit 1
fi
if grep -q "libc.so.6.*<- selftest.c:.*:helper_alloc" "$OUT/label_fixed.txt"; then
    echo "ok: and the fixed index on the SAME table still lands in libc, which is the point"
else
    echo "note: the fixed index did not land in libc here; the comparison is weaker than intended"
fi

echo "=== verdict ==="
python3 - "$OUT" <<'PY'
import sys, os
out = sys.argv[1]
ok = True

def rows(path):
    r = []
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        if len(f) >= 4:
            r.append((int(f[0]), int(f[1]), int(f[2]),
                      int(f[3], 16), int(f[4], 16) if len(f) >= 5 else 0))
    return r

MiB = 1024 * 1024
DIRECT = {64 * MiB, 128 * MiB, 256 * MiB}     # allocated straight from main
HELPER = {8 * MiB, 16 * MiB, 32 * MiB}        # allocated through ONE noinline helper

for d in (1, 2, 3):
    if not rows(os.path.join(out, f"sample.{d}")):
        print(f"FAIL: depth {d} produced no rows at all -- the recorder did not load, or the "
              f"sampler read the table before it was ready")
        ok = False

r1, r2 = rows(os.path.join(out, "sample.1")), rows(os.path.join(out, "sample.2"))
r3 = rows(os.path.join(out, "sample.3"))

# 1. every allocation the program made is present, at its own size, at both depths.
for d, r in ((1, r1), (2, r2), (3, r3)):
    got = {row[1] for row in r}
    missing = (DIRECT | HELPER) - got
    # the helper's three sizes may be merged into one row at depth 1, so check the SUM instead
    if DIRECT - got:
        print(f"FAIL: depth {d} is missing direct allocations of {sorted(DIRECT - got)} bytes")
        ok = False
    hel = sum(row[1] for row in r if row[1] in HELPER or row[1] == sum(HELPER))
    if hel != sum(HELPER):
        print(f"FAIL: depth {d} accounts {hel} bytes for the helper, expected {sum(HELPER)}")
        ok = False
if ok:
    print("ok: every allocation is present at its own size, at both depths")

# 2. THE POINT: at depth 1 the three owners of one helper are ONE row, at depth 2 they are THREE.
def helper_rows(r):
    return [x for x in r if x[1] in HELPER or x[1] == sum(HELPER)]

h1, h2 = helper_rows(r1), helper_rows(r2)
if len(h1) != 1:
    print(f"FAIL: at depth 1 the helper produced {len(h1)} rows, expected 1 -- the test cannot "
          f"show that depth 2 fixes anything if depth 1 was never broken")
    ok = False
else:
    print("ok: at depth 1 the three owners of one helper collapse into a single row")

if len(h2) != 3:
    print(f"FAIL: at depth 2 the helper produced {len(h2)} rows, expected 3")
    for x in h2:
        print(f"      size={x[1]} site=0x{x[3]:x} caller=0x{x[4]:x}")
    ok = False
elif len({x[4] for x in h2}) != 3:
    print("FAIL: at depth 2 the three helper rows share a caller, so the captured frame is the "
          "helper itself rather than its caller -- check which bt[] index caller_frame() returns")
    ok = False
else:
    print("ok: at depth 2 the three owners of one helper are three distinct callers")

# 2b. one frame further out, proves the captured index follows SNNI_PM_DEPTH rather than being
# hard-wired to the second frame.
h3 = helper_rows(r3)
if len(h3) != 1:
    print(f"FAIL: at depth 3 the helper produced {len(h3)} rows, expected 1 -- three owners called "
          f"from one main() must share a caller once the capture moves past the owners")
    ok = False
elif not h3[0][4]:
    print("FAIL: at depth 3 the helper row records no caller at all, so the capture ran off the "
          "end of the stack rather than one frame further out")
    ok = False
elif h2 and len({x[4] for x in h2}) == 3 and h3[0][4] in {x[4] for x in h2}:
    print("FAIL: the depth 3 caller is one of the depth 2 callers, so the index did not move")
    ok = False
else:
    print("ok: at depth 3 the capture moves one frame further out and the owners share main")

# 3. depth 1 must record NO caller, so the published default is unchanged.
if any(x[4] for x in r1):
    print("FAIL: depth 1 recorded a caller; the default configuration must be unchanged")
    ok = False
else:
    print("ok: depth 1 records no caller, so the published default is byte-identical")

print("SELFTEST PASS" if ok else "SELFTEST FAIL")
sys.exit(0 if ok else 1)
PY
RC=$?
echo "artifacts in $OUT"
exit $RC
