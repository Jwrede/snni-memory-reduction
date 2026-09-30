#!/bin/bash
# SHARK target micro-benchmark on server C (jwmaster01); the procedure of impl/floor_bench.sbatch.
# Binary: benchmark-shark_floor from the campaign image shark-s0_default (docker 2e4e40decd2b,
# SHARK 6957e5e, built from impl/src/shark_floor_bench.cpp, code identical to the repo file),
# copied out with its libraries (libstdc++, libgomp, libgcc_s, libc, libm) and the image's loader,
# and run on the host through that loader (rootless podman there cannot load the image).
# Dealer writes server.dat/client.dat, then party 0 and party 1 with SHARK_ONESHOT=0, OMP 4. 3 runs.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PKG=${PKG:-$HERE/pkg}                  # benchmark-shark_floor + lib/
RUN="$PKG/lib/ld-linux-x86-64.so.2 --library-path $PKG/lib $PKG/benchmark-shark_floor"
WORK=/data2/j_wred02_shark_floor
export OMP_NUM_THREADS=4
md5sum "$PKG/benchmark-shark_floor"
for i in 1 2 3; do
  OUT="$HERE/logs/run$i"; mkdir -p "$OUT"; rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
  {
    $RUN 2 > "$OUT/floor_dealer.out" 2> "$OUT/floor_dealer.err"; echo "dealer rc=$?"
    stat -c '%n %s' server.dat client.dat
    export SHARK_ONESHOT=0
    $RUN 0 > "$OUT/floor_party0.out" 2> "$OUT/floor_party0.err" & P0=$!
    sleep 3
    $RUN 1 > "$OUT/floor_party1.out" 2> "$OUT/floor_party1.err" & P1=$!
    wait $P0; echo "party0 rc=$?"; wait $P1; echo "party1 rc=$?"
  } > "$OUT/driver.log" 2>&1
  cd "$HERE"
  grep -q FLOORDONE "$OUT/floor_party0.err" && echo "run$i FLOOR_OK" || echo "run$i FAILED"
done
rm -rf "$WORK"
echo RUN_DONE
