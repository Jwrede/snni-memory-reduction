#!/bin/bash
# Build and run the MOAI bootstrap-instant target micro-benchmark on server C (jwmaster01), podman.
# Base image localhost/moai-stock: MOAI 8bcb0ea, SEAL fork installed, 2-layer loop edit only.
# `git stash` restores the pinned test_full_scheme.hpp, then impl/p_floor_bench_boot.py patches it.
# 3 runs at T=1 (SNNI_FLOOR_BOOT=1 OMP_NUM_THREADS=1); the binary prints FLOOR_BOOT| lines.
set -uo pipefail
CTR=moai-floor-boot
HERE=$(cd "$(dirname "$0")" && pwd)
podman rm -f $CTR >/dev/null 2>&1 || true
podman run --name $CTR -d --entrypoint sleep localhost/moai-stock:latest infinity
podman exec $CTR sh -c 'cd /root/moai && git stash && git rev-parse HEAD && git status --short'
# the image has no python3: patch a host copy of the header and copy it back
W=$(mktemp -d); mkdir -p "$W/include/test"
podman cp $CTR:/root/moai/include/test/test_full_scheme.hpp "$W/include/test/test_full_scheme.hpp"
python3 "$HERE/../../impl/p_floor_bench_boot.py" "$W" || exit 1   # repo layout; on server C it ran from a copy beside the script (same md5)
podman cp "$W/include/test/test_full_scheme.hpp" $CTR:/root/moai/include/test/test_full_scheme.hpp
podman exec $CTR sh -c 'cd /root/moai && git diff --stat && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j 16 2>&1 | tail -3 && strings -a build/test | grep -q "FLOOR_BOOT_SUMMARY|" && md5sum build/test'
rc=$?; echo "BUILD rc=$rc"; [ $rc -eq 0 ] || exit 1
mkdir -p "$HERE/logs"
for i in 1 2 3; do
  /usr/bin/time -v podman exec -w /root/moai/build -e SNNI_FLOOR_BOOT=1 -e OMP_NUM_THREADS=1 $CTR /root/moai/build/test > "$HERE/logs/run$i.out" 2>&1
done
podman rm -f $CTR >/dev/null 2>&1
echo RUN_DONE
