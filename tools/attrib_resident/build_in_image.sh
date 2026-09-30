#!/bin/bash
# Builds the resident recorder for the container it is preloaded into.
#
#   build_in_image.sh <image> [outdir]                  # build inside <image> (needs gcc there)
#   build_in_image.sh --builder <image> <target> [out]  # build inside a builder, for <target>
#
# Host toolchain not usable: a newer glibc than the target's makes LD_PRELOAD fail.
# One build against the oldest glibc (e.g. `--builder docker://gcc:9-buster`) covers all targets.
# With a target given, the target's loader is tested on the built library.
# pmsample is built for the poller's side: inside the container (podman), host (apptainer).
set -uo pipefail

BUILDER=""
if [ "${1:-}" = "--builder" ]; then BUILDER="$2"; shift 2; fi
IMAGE="${1:?Usage: build_in_image.sh [--builder <img>] <image> [outdir]}"
OUT="${2:-$(cd "$(dirname "$0")" && pwd)}"
BUILD_IN="${BUILDER:-$IMAGE}"

runner() {  # podman for local names, apptainer for sif paths and docker:// refs
    case "$1" in
        *.sif|docker://*|library://*) echo apptainer ;;
        *) echo podman ;;
    esac
}

do_exec() {  # <image> <script>
    if [ "$(runner "$1")" = apptainer ]; then
        apptainer exec --bind "$OUT":/ar "$1" bash -c "$2"
    else
        podman run --rm -v "$OUT":/ar --entrypoint bash "$1" -c "$2"
    fi
}

do_exec "$BUILD_IN" 'set -e; cd /ar
  gcc -O2 -g -shared -fPIC -o pmrec.so pmrec.c -ldl -lpthread
  gcc -O2 -g -o pmsample pmsample.c' || exit 1

echo "built pmrec.so and pmsample in $BUILD_IN"
do_exec "$BUILD_IN" 'objdump -T /ar/pmrec.so | grep -o "GLIBC_[0-9.]*" | sort -u | tr "\n" " "'
echo "<- symbol versions pmrec.so requires"

# Proof, not inference: the target's loader is asked directly to accept the library.
if [ -n "$BUILDER" ]; then
    if do_exec "$IMAGE" 'LD_PRELOAD=/ar/pmrec.so /bin/echo loader-accepted-it' ; then
        echo "OK: $IMAGE loads the recorder"
    else
        echo "ERROR: $IMAGE refuses the recorder; pick an older builder image" >&2
        exit 1
    fi
fi
