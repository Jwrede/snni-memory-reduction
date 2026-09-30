#!/bin/bash
#SBATCH --job-name=bbbuild
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/bumblebee/build-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/bumblebee/build-%j.out
#
# Builds one BumbleBee step image (as a job, not on the login node).
#
#   sbatch --export=ALL,STEP=b0_default build_step.sbatch
#   sbatch --export=ALL,STEP=b1_xxx,PARENT=b0_default build_step.sbatch
#
# Step = overlay directory $BASE/build/<step>/tree/ mirroring the OpenBumbleBee tree; %post copies it
# over the checkout and re-runs bazel. Bazel disk cache bind-mounted (image stays a few hundred MB).
set -uo pipefail

STEP="${STEP:?need STEP, e.g. STEP=b0_default}"
PARENT="${PARENT:-}"

C=/scratch/tmp/j_wred02/snni_campaign
B=$C/bumblebee
OVERLAY=$B/build/$STEP
OUT=$B/sif/bumblebee-$STEP.sif

# Pinned. A moving upstream would make two steps of one waterfall two different programs.
OBB_COMMIT=47c5560d069543591f3b0176eb530ede28e33fc3

mkdir -p "$B/sif" "$B/def" "$B/bz" "$B/apptmp" || exit 2
[ -d "$OVERLAY/tree" ] || { echo "no overlay at $OVERLAY/tree" >&2; exit 2; }

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }

export APPTAINER_TMPDIR=$B/apptmp
export APPTAINER_CACHEDIR=$B/apptmp/cache
mkdir -p "$APPTAINER_CACHEDIR"

# Not --exclusive: a build isn't a measurement, so it doesn't need a whole node.
# Half the allocated cores: these are debug-info C++ TUs of SEAL/SPU, and an OOM after 40 min
# costs more than 10 extra minutes; bazel is additionally capped by --local_ram_resources below.
JOBS=$(( $(nproc) / 2 ))
[ "$JOBS" -lt 1 ] && JOBS=1

apptainer build --help 2>&1 | grep -q -- '--bind' \
    || echo "WARNING: this apptainer build does not advertise --bind; the disk cache may not mount"

def=$B/def/$STEP.def
cat > "$def" <<EOF
Bootstrap: docker
From: ubuntu:22.04

# BumbleBee step $STEP. OpenBumbleBee pinned at $OBB_COMMIT.
#
# /opt, not /tmp: apptainer bind-mounts the host /tmp over the container's during %post, so a file
# staged into /tmp by %files is invisible to the very section that consumes it.

%files
    $OVERLAY/tree /opt/overlay

%post
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y gcc-11 g++-11 libasan6 git wget curl unzip autoconf make \\
        lld-15 cmake ninja-build nasm procps binutils python3 python3-dev python3-pip
    apt-get clean
    ln -sf /usr/bin/python3 /usr/bin/python
    pip3 install --no-cache-dir numpy
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
    update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-15 100

    wget -qO /usr/bin/bazel \\
        https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
    chmod +x /usr/bin/bazel

    cd /root
    git clone https://github.com/AntCPLab/OpenBumbleBee.git
    cd OpenBumbleBee
    git checkout $OBB_COMMIT

    # THE STEP'S OWN SOURCE. Paths in the overlay mirror the tree, so this is the whole of what
    # makes this step different from its parent.
    test -d /opt/overlay
    cp -rv /opt/overlay/. /root/OpenBumbleBee/
    rm -rf /opt/overlay

    # -g, full debug info, and NOT -g1. This was measured rather than assumed: a first build used
    # -g1 on the argument that line tables are all a symboliser reads, and the resulting baseline
    # named its dominant 6.5 GB object
    #     int128.cc:?:base_of_encoded_value(unsigned char, _Unwind_Context*) [clone .cold]
    # which is a libgcc unwinder symbol with no line, i.e. nearest-symbol garbage, for an address
    # that lies in this very binary. That is exactly the failure PROCEDURE.md describes: without
    # real debug info a symboliser returns chains that read like call stacks and are not.
    # Debug sections are not resident, so this costs image size and nothing else.
    # --strip=never so that the linker keeps what the compiler emitted.
    # LINK libstdc++ DYNAMICALLY, overriding SPU's own .bazelrc, and this is a measurement
    # requirement rather than a preference. SPU sets
    #     build:linux --action_env=BAZEL_LINKOPTS=-static-libstdc++:-static-libgcc
    #     build:linux --action_env=BAZEL_LINKLIBS=-l%:libstdc++.a:-l%:libgcc.a
    # so `operator new` is DEFINED INSIDE the binary and every C++ allocation is bound to it at link
    # time. LD_PRELOAD cannot interpose on that, so the campaign's recorder never sees the call
    # sites: it catches the malloc inside the internal operator new instead, one frame too deep and
    # at the SAME address for every allocation in the program.
    #
    # Measured, before this override: 6.10 GB of a 6.66 GB attributed total sat on one site that
    # resolves to `operator new(unsigned long)` with 161 allocations. That is not an object, it is
    # "every C++ allocation", and it cannot be a step target any more than an `anon:` entry can.
    # It is the exact failure PROCEDURE.md records ("C++ hides call sites behind operator new ...
    # 956 allocations collapsed onto one site"), reached by a different route: static linkage rather
    # than a malloc wrapper.
    #
    # `-lstdc++:-lm` and not an empty BAZEL_LINKLIBS: empty removes the C++ runtime from the link
    # line altogether and the link fails on std::cerr, __cxa_end_catch and friends. This is bazel's
    # own unix default, i.e. dynamic linkage.
    #
    # Dynamic linkage changes where libstdc++'s text lives, not what the protocol allocates.
    bazel build -c opt --copt=-g --strip=never \\
        --action_env=BAZEL_LINKOPTS= --action_env=BAZEL_LINKLIBS=-lstdc++:-lm \\
        --disk_cache=/mnt/cache --repository_cache=/mnt/repo \\
        --jobs=$JOBS --local_ram_resources=HOST_RAM*.7 \\
        //examples/cpp/bert_bench:op_bench

    mkdir -p /opt/bumblebee
    cp -L bazel-bin/examples/cpp/bert_bench/op_bench /opt/bumblebee/op_bench
    chmod +x /opt/bumblebee/op_bench
    # Keep the source that produced the binary IN the image, so a check of what this step ran does
    # not depend on any file outside it.
    mkdir -p /opt/bumblebee/src
    cp -r /root/OpenBumbleBee/examples/cpp/bert_bench /opt/bumblebee/src/
    cp -r /root/OpenBumbleBee/libspu/mpc/cheetah /opt/bumblebee/src/cheetah

    bazel shutdown || true
    # The bazel OUTPUT BASE is scratch, not artifact: it is tens of GB of intermediate objects, and
    # every step image would otherwise carry a copy. The reusable part of it is already in the
    # bind-mounted disk cache, which lives outside the image.
    # THE CLEANUP MUST NOT BE ABLE TO FAIL THE BUILD. It once did, and it cost a 15-minute job
    # whose compile and link had already SUCCEEDED: bazel's foreign_cc rules leave openssl's copied
    # outputs read-only, `rm -rf` under fakeroot reports "Permission denied" on them, and `set -e`
    # then aborts %post after the binary was already in place. Make the tree writable first and
    # swallow the result: nothing here produces the artifact, it only makes it smaller.
    chmod -R u+w /root/.cache/bazel 2>/dev/null || true
    rm -rf /root/.cache/bazel /root/OpenBumbleBee/bazel-* /root/.cache/bazelisk 2>/dev/null || true
    rm -rf /root/OpenBumbleBee/.git 2>/dev/null || true
    rm -rf /var/lib/apt/lists/* 2>/dev/null || true
    # The one thing in this section that IS load-bearing: the binary has to exist.
    test -x /opt/bumblebee/op_bench

%labels
    campaign snni-memory-reduction
    system bumblebee
    step $STEP
    upstream OpenBumbleBee@$OBB_COMMIT
EOF

echo "=== building $OUT with $JOBS jobs"
date -u +%FT%TZ
# --force so a rebuild replaces the image rather than refusing on a prior attempt's artifact.
# Bound onto /mnt (exists in the base image; a nonexistent bind destination fails the build).
apptainer build --force --fakeroot \
    --bind "$B/bz":/mnt \
    "$OUT" "$def" || { echo "BUILD_FAIL: apptainer build"; exit 1; }
date -u +%FT%TZ

# The artifact is verified, not assumed.
BIN=/opt/bumblebee/op_bench
NEW=$(apptainer exec "$OUT" md5sum $BIN | awk '{print $1}')
echo "binary md5 = $NEW"
echo "$NEW" > "$B/sif/bumblebee-$STEP.binary_md5"

# The gate observable must be in the binary, checked cheaply here rather than after three
# replicates, via the format strings `strings` can see (keep patterns in step with op_bench.cc).
# The recorder must also be able to see `operator new`: if it's defined in the binary rather than
# imported, LD_PRELOAD can't interpose and the object table silently collapses onto one site.
apptainer exec "$OUT" bash -c "readelf -d $BIN | grep -q 'libstdc++'" \
    || { echo "BUILD_FAIL: libstdc++ is not a dynamic dependency, so operator new cannot be"
         echo "            interposed and the host decomposition would name nothing."; exit 7; }
apptainer exec "$OUT" bash -c "nm -D $BIN 2>/dev/null | grep -q ' U _Znwm'" \
    || { echo "BUILD_FAIL: _Znwm (operator new) is not an UNDEFINED symbol in this binary"; exit 7; }
echo "operator new is imported, so the recorder can intercept it"

for lit in 'GATE|%s|rank=0|n=%zu|file=%s' 'GATESUM|%s|rank=%d' 'COMM|rank=%d' 'SEEDED|mt19937'; do
    apptainer exec "$OUT" bash -c "strings $BIN | grep -qF '$lit'" \
        || { echo "BUILD_FAIL: '$lit' is absent from this binary"; exit 5; }
done
echo "gate, checksum, transcript and setup markers all present in the binary"

# A step image can contain the right source and the wrong binary: if identical to the parent's, the overlay never reached the compiler.
if [ -n "$PARENT" ]; then
    POLD=$B/sif/bumblebee-$PARENT.binary_md5
    [ -r "$POLD" ] || { echo "BUILD_FAIL: no recorded md5 for parent $PARENT"; exit 6; }
    OLD=$(cat "$POLD")
    echo "parent ($PARENT) binary md5 = $OLD"
    [ "$NEW" != "$OLD" ] || { echo "BUILD_FAIL: binary identical to parent; the overlay did not take"; exit 6; }
fi

ls -la "$OUT"
echo "BUILD_OK"
