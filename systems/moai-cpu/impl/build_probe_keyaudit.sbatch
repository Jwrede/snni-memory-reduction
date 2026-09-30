#!/bin/bash
#SBATCH --job-name=moaicpu-pimg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/pimg-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/moai-cpu/pimg-%j.out

# Key-audit probe image: baseline image + p_keyaudit.py, recompiled (no re-clone from the .def).
# Not for waterfall rows (uncosted probe instrumentation).
set -uo pipefail

B=/scratch/tmp/j_wred02/snni_campaign/moai-cpu
PARENT="$B/sif/moai-cpu-m0_default.sif"
OUT="${OUT:-$B/sif/moai-cpu-p_keyaudit.sif}"

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH" >&2; exit 2; }
[ -r "$PARENT" ] || { echo "FAILED: no parent image at $PARENT" >&2; exit 2; }
[ -r "$B/p_keyaudit.py" ] || { echo "FAILED: no patch script at $B/p_keyaudit.py" >&2; exit 2; }

def="$B/def/$(basename "$OUT" .sif).def"
mkdir -p "$B/def"
cat > "$def" <<EOF
Bootstrap: localimage
From: $PARENT

%files
    $B/p_keyaudit.py /opt/p_keyaudit.py

%post
    set -e
    cd /root/moai
    python3 /opt/p_keyaudit.py /root/moai
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j "\$(nproc)"

    # Same rule as the baseline image: the binary must carry what the source promised, checked
    # against the ARTIFACT rather than against the patch script's own report.
    strings -a /root/moai/build/test | grep -q 'KEYAUDIT|' \
        || { echo "FAILED: no key-audit marker in the binary"; exit 4; }
    strings -a /root/moai/build/test | grep -q 'POOL|' \
        || { echo "FAILED: no pool marker in the binary"; exit 4; }
    strings -a /root/moai/build/test | grep -q 'SEEDED|seal_prng|' \
        || { echo "FAILED: the baseline's seed marker was lost"; exit 4; }
    strings -a /root/moai/build/test | grep -q 'MOAI_NUM_X' \
        || { echo "FAILED: no batch knob in the binary"; exit 4; }
    md5sum /root/moai/build/test > /opt/moai_binary_md5.txt
    cat /opt/moai_binary_md5.txt
EOF

echo "=== building $OUT"
apptainer build --force --fakeroot "$OUT" "$def" || { echo "BUILD_FAIL"; exit 1; }

BIN=/root/moai/build/test
NEW=$(apptainer exec "$OUT" md5sum $BIN | awk '{print $1}')
OLD=$(apptainer exec "$PARENT" md5sum $BIN | awk '{print $1}')
echo "baseline binary md5 = $OLD"
echo "probe    binary md5 = $NEW"
[ "$NEW" != "$OLD" ] || { echo "BUILD_FAIL: binary identical to the baseline, the patch did not take"; exit 3; }
echo "BUILD_OK"
