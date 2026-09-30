#!/bin/bash
#SBATCH --job-name=pmbuild
#SBATCH --partition=express
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/j_wred02/snni_campaign/pmbuild-%j.out
#SBATCH --error=/scratch/tmp/j_wred02/snni_campaign/pmbuild-%j.out
#
# Builds the resident recorder for the LLAMA dealer line (as a job, not on the login node).
# pmrec.so: container glibc (LD_PRELOAD). pmsample: host glibc (runs beside the poller).
# Wrong libc: `GLIBC_2.xx not found`, every run exits immediately.
set -uo pipefail

B=/scratch/tmp/j_wred02/snni_campaign/llama-dealer
AR=/scratch/tmp/j_wred02/snni_campaign/attrib_resident
SIF="$B/sif/llama-dealer-d0_default.sif"

module purge 2>/dev/null
module load Apptainer/1.5.0 2>/dev/null
command -v apptainer >/dev/null || { echo "FAILED: apptainer not on PATH"; exit 1; }
[ -f "$SIF" ] || { echo "FAILED: $SIF missing"; exit 1; }

echo "=== pmrec.so against the CONTAINER glibc"
apptainer exec --bind "$AR":/ar "$SIF" \
    bash -c 'cd /ar && gcc -O2 -g -shared -fPIC -o pmrec.so pmrec.c -ldl -lpthread' \
    || { echo "FAILED: pmrec.so did not build"; exit 1; }

echo "=== pmsample against the HOST glibc"
( cd "$AR" && gcc -O2 -g -o pmsample pmsample.c ) \
    || { echo "FAILED: pmsample did not build"; exit 1; }

echo "=== what each was linked against"
# --bind is required: without it $AR is not inside the container, objdump silently finds no file
# and the GLIBC line comes out empty, which reads like "requires nothing" rather than "was not
# looked at".
apptainer exec --bind "$AR":/ar "$SIF" objdump -T /ar/pmrec.so \
    | grep -o 'GLIBC_[0-9.]*' | sort -u | tr '\n' ' '
echo "(container, for pmrec.so)"
objdump -T "$AR/pmsample" | grep -o 'GLIBC_[0-9.]*' | sort -u | tr '\n' ' '
echo "(host, for pmsample)"

# Proves the container's loader accepts the preload (trivial program) before any measurement
# depends on it; "worked on another image" has failed before.
echo "=== does the container loader accept the preload?"
apptainer exec --bind "$AR":/ar "$SIF" \
    env LD_PRELOAD=/ar/pmrec.so SNNI_PM_TABLE=/tmp/pmtable_probe SNNI_PM_MIN=65536 \
    /bin/echo loader-accepted-it \
    || { echo "FAILED: the container refuses pmrec.so"; exit 1; }

# Checks the sampler runs on this host. Captured into a variable, not piped: pmsample's usage
# exit code (2) would fail the check under pipefail regardless of what grep found.
probe=$("$AR/pmsample" 2>&1 || true)
case "$probe" in
    *usage*) echo "pmsample runs on the host" ;;
    *) echo "FAILED: pmsample does not execute on the host: $probe"; exit 1 ;;
esac

ls -l "$AR"
echo "RECORDER_OK"
