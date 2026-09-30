#!/bin/bash
# On PALMA: docker archive -> one .sif per step; resident recorder built against two libcs.
#   build_sifs.sh [step ...]
#   pmrec.so   is LD_PRELOADed into the container's process -> container glibc
#   pmsample   runs beside the poller, on the host          -> host glibc
# Wrong libc: `version 'GLIBC_2.xx' not found`, every run exits in zero seconds.
set -uo pipefail

BASE=/scratch/tmp/j_wred02/snni_campaign/llama-dealer
AR=/scratch/tmp/j_wred02/snni_campaign/attrib_resident
TAR="$BASE/images/llama-dealer-steps.tar"
STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || STEPS=(d0_default d1_select_free d2_mask_stream d3_activation_free
                                 d4_truncate_free d5_gelu_free)

module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
mkdir -p "$BASE/sif" "$AR"
[ -f "$TAR" ] || { echo "ERROR: $TAR missing; run export_images.sh first" >&2; exit 2; }

for s in "${STEPS[@]}"; do
    out="$BASE/sif/llama-dealer-$s.sif"
    [ -f "$out" ] && { echo "have $out"; continue; }
    echo "=== building $out"
    apptainer build "$out" "docker-archive://$TAR:localhost/llama-dealer-$s:latest" || exit 1
done

echo "=== recorder: pmrec.so against the CONTAINER glibc"
apptainer exec --bind "$AR":/ar "$BASE/sif/llama-dealer-${STEPS[0]}.sif" \
    bash -c 'cd /ar && gcc -O2 -g -shared -fPIC -o pmrec.so pmrec.c -ldl -lpthread' || exit 1

echo "=== sampler: pmsample against the HOST glibc"
( cd "$AR" && gcc -O2 -g -o pmsample pmsample.c ) || exit 1

echo "=== what each was linked against:"
apptainer exec "$BASE/sif/llama-dealer-${STEPS[0]}.sif" \
    objdump -T "$AR/pmrec.so" | grep -o 'GLIBC_[0-9.]*' | sort -u | tr '\n' ' '
echo "(container, for pmrec.so)"
objdump -T "$AR/pmsample" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -u | tr '\n' ' '
echo "(host, for pmsample)"

ls -l "$BASE/sif/"
for s in "${STEPS[@]}"; do
    printf '%-28s binary_md5=%s\n' "$s" \
        "$(apptainer exec "$BASE/sif/llama-dealer-$s.sif" \
             md5sum /opt/EzPC/GPU-MPC/ext/sytorch/build/bertbenchmark | awk '{print $1}')"
done
