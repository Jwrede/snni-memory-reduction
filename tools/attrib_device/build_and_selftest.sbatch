#!/bin/bash
#SBATCH --job-name=cudarec-selftest
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/diag/cudarec-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-gpu/diag/cudarec-%j.err
#
# Builds cudarec.so inside the target image (host toolchain may be too new for the container's
# loader) and runs selftest.cu, which checks the recorded byte counts.
set -uo pipefail

BASE=/scratch/tmp/j_wred02/snni_campaign/moai-gpu
DIAG=$BASE/diag
# Must test the .so the measurements actually load ($CAMP/attrib_device), not a private diag/
# copy: until 2026-08-08 the test proved a freshly built recorder while measurements loaded one
# eight days old.
AD=/scratch/tmp/j_wred02/snni_campaign/attrib_device
SIF=/scratch/tmp/j_wred02/moai_gpu/moai-gpu-s0_default.sif
OUT=$DIAG/cudarec_selftest
rm -rf "$OUT"; mkdir -p "$OUT" || exit 2

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null

RC=0
apptainer exec --nv --bind /scratch:/scratch "$SIF" bash -lc "
    set -e
    cd $AD
    echo '=== build cudarec.so ==='
    gcc -O2 -g -fPIC -shared -o cudarec.so cudarec.c -ldl
    ls -l cudarec.so

    echo '=== build the selftest ==='
    # --default-stream per-thread matches the measured binary's _ptsz interposition; -cudart=shared
    # is required since nvcc statically links cudart by default (untestable otherwise); -fopenmp tests thread safety.
    /usr/local/cuda-11.8/bin/nvcc -O2 -g --default-stream per-thread -cudart=shared \
        -Xcompiler -fopenmp -o selftest selftest.cu

    echo '=== which symbols does the selftest import? ==='
    nm -D -u ./selftest | grep -iE 'cudaMalloc|cudaFree' || echo '(none)'
    # A selftest whose allocation calls are not interposable cannot test the recorder, and would
    # report a clean zero that reads like a broken recorder rather than a broken test.
    if ! nm -D -u ./selftest | grep -qi 'cudaMallocAsync'; then
        echo 'FAILED: the selftest does not import cudaMallocAsync, so nothing can interpose it.' >&2
        echo '        Check -cudart=shared. This is a defect in the TEST, not in cudarec.' >&2
        exit 3
    fi

    echo '=== run WITHOUT the recorder (must produce no table) ==='
    cd $OUT && $AD/selftest

    echo '=== run WITH the recorder (depth 1, the published default) ==='
    cd $OUT
    SNNI_CUDAREC=1 SNNI_CUDAREC_OUT=$OUT LD_PRELOAD=$AD/cudarec.so $AD/selftest

    echo '=== run WITH the recorder at DEPTH 2 (caller capture) ==='
    mkdir -p $OUT/d2
    SNNI_CUDAREC=1 SNNI_CUDAREC_DEPTH=2 SNNI_CUDAREC_OUT=$OUT/d2 \
        LD_PRELOAD=$AD/cudarec.so $AD/selftest
" 2>&1
RC=$?

echo
echo "=== the table ==="
cat "$OUT/device_table.txt" 2>/dev/null || { echo "NO TABLE WRITTEN"; RC=9; }

echo
echo "=== verdict ==="
python3 - "$OUT/device_table.txt" <<'PY'
import sys, re
try:
    txt = open(sys.argv[1]).read()
except OSError:
    print("FAIL: no device_table.txt"); sys.exit(9)

EXPECT = 454 * 1024 * 1024
m = re.search(r"peak_device_bytes=(\d+)", txt)
peak = int(m.group(1)) if m else -1
# every key=value in EVERY header line, not just the first: the hook counts are on line 1 and
# untracked_frees / stale_reuse are on line 2, and reading only line 1 made both read as 0.
hooks = dict(re.findall(r"(\w+)=(\d+)",
                        "\n".join(l for l in txt.splitlines() if l.startswith("#"))))
rows = [l for l in txt.splitlines() if l and not l.startswith("#")]

ok = True
if peak != EXPECT:
    print(f"FAIL: peak_device_bytes={peak}, expected {EXPECT}")
    ok = False
else:
    print(f"ok: peak_device_bytes={peak} = 454 MiB exactly")

# 4 in phase 1, then the churn: N initial + rounds x N/2 replacements, N = 40000, rounds = 3.
EXPECT_ALLOCS = 4 + 40000 + 3 * 20000 + 8 * 2000 * 4 + 3
n_ptsz = int(hooks.get("malloc_async_ptsz", 0))
if n_ptsz != EXPECT_ALLOCS:
    print(f"FAIL: malloc_async_ptsz fired {n_ptsz} times, expected {EXPECT_ALLOCS}")
    ok = False
else:
    print(f"ok: the _ptsz hook fired {n_ptsz} times, which is every allocation the program made")

# Strongest invariant: since every pointer is freed, the recorder's live total must return to
# 0; a lost update leaves a residue (this is what four negative sites in a real run looked like).
live_exit = int(hooks.get("live_at_exit", -1))
if live_exit != 0:
    print(f"FAIL: live_at_exit={live_exit}, expected 0 -- every pointer was freed, so a nonzero "
          f"residue is an update the recorder lost")
    ok = False
else:
    print("ok: live_at_exit=0, so no accounting update was lost across 8 concurrent threads")

# Churn invariants (what the 2026-08-08 pointer-table fix addressed): without tombstones a
# free can miss its entry, get reused, and the next free corrupts the wrong site's balance.
n_untracked = int(hooks.get("untracked_frees", 0))
n_stale = int(hooks.get("stale_reuse", 0))
if n_untracked or n_stale:
    print(f"FAIL: untracked_frees={n_untracked}, stale_reuse={n_stale}; both must be 0 here "
          f"because this program frees every pointer it allocated, after the recorder was armed")
    ok = False
else:
    print("ok: no untracked free and no stale reuse over 100,004 allocations")

neg = [l for l in rows if int(l.split()[0]) >= (1 << 63) or int(l.split()[2]) >= (1 << 63)]
if neg:
    print(f"FAIL: {len(neg)} site(s) have a wrapped (negative) balance:")
    for l in neg[:5]:
        print("      " + l)
    ok = False
else:
    print("ok: no site has a negative balance")

if int(hooks.get("malloc_async", 0)) != 0:
    print("note: the non-ptsz hook also fired; harmless, but the binary was expected to use _ptsz only")

if len(rows) < 3:
    print(f"FAIL: {len(rows)} call sites in the table, expected at least 3")
    ok = False
else:
    print(f"ok: {len(rows)} distinct call sites, so sites are not collapsing onto one")

tot = sum(int(l.split()[0]) for l in rows)
if tot != EXPECT:
    print(f"FAIL: site bytes sum to {tot}, expected {EXPECT} -- the per-site snapshot and the "
          f"peak disagree, which would make every object table wrong")
    ok = False
else:
    print("ok: the per-site snapshot sums to the peak, so the aggregate describes the peak instant")

# THE CALLER CAPTURE, tested so that depth 1 cannot pass it: three owners allocate through ONE
# noinline helper, so depth 1 must show them as ONE site and depth 2 as THREE.
import os
d2 = os.path.join(os.path.dirname(sys.argv[1]), "d2", "device_table.txt")

def helper_sites(path, sizes=(1 << 20, 2 << 20, 3 << 20)):
    """Rows whose byte total is one of the helper's three sizes and whose count is 1."""
    try:
        txt2 = open(path).read()
    except OSError:
        return None
    out = []
    for l in txt2.splitlines():
        if l.startswith("#") or not l.strip():
            continue
        f2 = l.split()
        if len(f2) >= 4 and int(f2[2]) == 1 and int(f2[0]) in sizes:
            out.append(l)
    return out

h1 = helper_sites(sys.argv[1])
h2 = helper_sites(d2)
if h2 is None:
    print("FAIL: no depth-2 table was written")
    ok = False
elif len(h2) != 3:
    print(f"FAIL: at depth 2 the three owners of one helper produced {len(h2)} site(s), expected 3")
    for l in h2:
        print("      " + l)
    ok = False
else:
    print("ok: at depth 2 the three owners of one helper are three distinct sites")

print("SELFTEST PASS" if ok else "SELFTEST FAIL")
sys.exit(0 if ok else 1)
PY
V=$?
echo "build_rc=$RC selftest_rc=$V"
[ "$RC" -eq 0 ] && [ "$V" -eq 0 ] || exit 1
exit 0
