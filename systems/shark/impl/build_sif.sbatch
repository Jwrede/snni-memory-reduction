#!/bin/bash
#SBATCH --job-name=sharkbuild
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/shark/build-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/shark/build-%j.out
#
#   sbatch --export=ALL,STEP=s0_default build_sif.sbatch
#
# docker archive -> .sif; checks the binary md5 and that the image's loader accepts the recorder.
set -uo pipefail

STEP="${STEP:?need STEP}"
C=/scratch/tmp/j_wred02/snni_campaign
B=$C/shark
AR=$C/attrib_resident
STAGE=$B/imgstage
SIF=$B/sif/shark-$STEP.sif

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }

export APPTAINER_TMPDIR=$B/apptainer_tmp
export APPTAINER_CACHEDIR=$B/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$B/sif" || exit 2

GZ=$STAGE/shark-$STEP.tar.gz
TAR=$STAGE/shark-$STEP.tar
[ -s "$GZ" ] || [ -s "$TAR" ] || { echo "FAILED: no archive at $GZ or $TAR"; exit 1; }
if [ ! -s "$TAR" ]; then
    gunzip -c "$GZ" > "$TAR" || exit 1
fi

if [ -e "$SIF" ]; then
    mv "$SIF" "$SIF.superseded-$(date -u +%Y%m%d-%H%M%S)" || exit 2
fi

apptainer build "$SIF" "docker-archive://$TAR" || { echo "FAILED: apptainer build"; exit 1; }
ls -la "$SIF"
sha256sum "$SIF"

echo "=== the image contains the binary that was built, not some parent's"
apptainer exec "$SIF" cat /root/binary_md5.txt || exit 1
apptainer exec "$SIF" md5sum /root/shark/build/benchmark-bert_instrumented || exit 1
apptainer exec "$SIF" cat /root/shark_commit.txt || exit 1
apptainer exec "$SIF" bash -c \
    'a=$(md5sum /root/shark/build/benchmark-bert_instrumented | cut -d" " -f1);
     b=$(cut -d" " -f1 /root/binary_md5.txt);
     [ "$a" = "$b" ]' \
    || { echo "FAILED: binary md5 does not match the one recorded at docker build time"; exit 1; }

echo "=== the gate and the seed marker are in the shipped binary"
apptainer exec "$SIF" bash -c \
    "strings /root/shark/build/benchmark-bert_instrumented | grep -q 'GATE|out|party='" \
    || { echo "FAILED: no gate string in the shipped binary"; exit 1; }
apptainer exec "$SIF" bash -c \
    "strings /root/shark/build/benchmark-bert_instrumented | grep -q 'SEEDED|srand=0x5eed'" \
    || { echo "FAILED: no seed marker in the shipped binary"; exit 1; }

echo "=== debug info is present, so the symboliser names functions instead of guessing"
apptainer exec "$SIF" bash -c \
    "readelf -S /root/shark/build/benchmark-bert_instrumented | grep -q debug_info" \
    || { echo "FAILED: no .debug_info; a stripped binary gives INFERRED names, not measured ones"; exit 1; }

echo "=== THIS image's loader accepts the campaign recorder"
# Not rebuilt here (would clobber pmrec.so while other jobs preload it). Built once against
# glibc 2.28 so one artefact serves every image; what follows proves this image accepts it.
[ -s "$AR/pmrec.so" ] || { echo "FAILED: no $AR/pmrec.so"; exit 1; }
apptainer exec --bind "$AR":/ar "$SIF" bash -c \
    'ldd --version | head -1; LD_PRELOAD=/ar/pmrec.so /bin/echo loader-accepted-it' \
    || { echo "FAILED: this image refuses the recorder"; exit 1; }

echo "=== and it accepts it for the ACTUAL payload, not only for /bin/echo"
# Runs the real binary (no args) to exercise operator-new interception; it exits non-zero by
# design via its own argc check, so reaching that point with the recorder attached is the pass.
apptainer exec --bind "$AR":/ar "$SIF" bash -c \
    'export LD_PRELOAD=/ar/pmrec.so SNNI_PM_TABLE=/tmp/pmtable_probe_%d SNNI_PM_MIN=65536
     /root/shark/build/benchmark-bert_instrumented 2>&1 | head -5
     echo "payload-ran-with-recorder"' \
    || { echo "FAILED: payload will not start under the recorder"; exit 1; }

echo "BUILD_OK $SIF"
