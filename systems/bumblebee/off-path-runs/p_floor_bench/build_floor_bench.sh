#!/bin/bash
# Build floor_bench inside the BumbleBee image on server C (jwmaster01).
# Base: localhost/bb-fm64-splits-full (OpenBumbleBee 47c5560, full LPN n=10,485,760, bazel cache).
# The image carries the splits-experiment edits in libspu/mpc/cheetah/{arith,state.h}; `git stash`
# restores the pinned commit's sources before the build, so the benchmark links stock libraries.
set -uo pipefail
CTR=bb-floor-bench
SRC=/root/OpenBumbleBee
HERE=$(cd "$(dirname "$0")" && pwd)
podman rm -f $CTR >/dev/null 2>&1 || true
podman run --name $CTR -d --network host --entrypoint sleep localhost/bb-fm64-splits-full:latest infinity
podman exec $CTR sh -c "cd $SRC && git stash && git status --short && git rev-parse HEAD && mkdir -p examples/cpp/floor_bench"
podman cp "$HERE/floor_bench.cc" $CTR:$SRC/examples/cpp/floor_bench/floor_bench.cc
podman cp "$HERE/BUILD.bazel"    $CTR:$SRC/examples/cpp/floor_bench/BUILD.bazel
podman exec $CTR sh -c "cd $SRC && grep -nE 'lpn_param_\{|kPolyDegree = ' libspu/mpc/cheetah/ot/yacl/yacl_ote_adapter.h libspu/mpc/cheetah/arith/cheetah_mul.cc && bazel build -c opt //examples/cpp/floor_bench:floor_bench"
rc=$?
echo "BUILD rc=$rc"
podman cp $CTR:$SRC/bazel-bin/examples/cpp/floor_bench/floor_bench "$HERE/floor_bench.bin" && md5sum "$HERE/floor_bench.bin"
