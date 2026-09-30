#!/bin/bash
# Rebuild a step's binary after its overlay was copied in, and prove the overlay changed it.
# COPY preserves the host mtime, so make can emit the parent's binary silently; touch the tree to
# force an unconditional rebuild, and fail the build if the new binary md5 equals the parent's.
set -euo pipefail
S=/opt/EzPC/GPU-MPC/ext/sytorch
PARENT=$(cat /opt/PARENT_BINARY_MD5)

find "$S" \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) -exec touch {} +

cd "$S/build"
make -j"$(nproc)" bertbenchmark > /tmp/build.log 2>&1 || { tail -30 /tmp/build.log; exit 3; }

NEW=$(md5sum bertbenchmark | awk '{print $1}')
if [ "$NEW" = "$PARENT" ]; then
    echo "ERROR: this step's binary is byte-identical to its parent's ($NEW)."
    echo "The overlay did not change the build, so this step cannot measure a lever."
    exit 4
fi
echo "$NEW" > /opt/PARENT_BINARY_MD5
echo "rebuilt ok: parent=$PARENT new=$NEW"
