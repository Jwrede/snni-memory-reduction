#!/bin/bash
# Run floor_bench in the build container: 3 x HE mode, 3 x OT mode (two processes, one per party).
set -uo pipefail
CTR=bb-floor-bench
B=/root/OpenBumbleBee/bazel-bin/examples/cpp/floor_bench/floor_bench
HERE=$(cd "$(dirname "$0")" && pwd)
ENVS="-e OMP_NUM_THREADS=1"
for i in 1 2 3; do
  podman exec $ENVS $CTR $B --mode he > "$HERE/he_run$i.out" 2>&1
  podman exec $ENVS $CTR $B --mode he_stock > "$HERE/he_stock_run$i.out" 2>&1
  podman exec $ENVS $CTR sh -c "$B --mode ot --rank 1 > /tmp/ot1.out 2>&1 & $B --mode ot --rank 0 > /tmp/ot0.out 2>&1; wait"
  podman cp $CTR:/tmp/ot0.out "$HERE/ot_run${i}_rank0.out"
  podman cp $CTR:/tmp/ot1.out "$HERE/ot_run${i}_rank1.out"
done
echo RUN_DONE
