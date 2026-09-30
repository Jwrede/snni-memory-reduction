#!/bin/bash
# Verify each step's library still contains its predecessors' cumulative patches. Dockerfile.step
# rebuilds from scratch with ONE EXTRA_PATCH, so a dropped patch silently regresses a step to a
# predecessor's peak (s5_up_split once measured s3's peak to a promille). Fingerprints the two lever
# library files read out of each image; fails only on a fingerprint that regresses to an earlier value.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SYS="$(dirname "$HERE")"
REMOTE="${1:-palma}"
B=/scratch/tmp/j_wred02/snni_campaign/shark

# Only the published line (steps/, from make_steps.py). off-path-runs/ are branches off the same
# predecessor, not successors, so their libraries do not form a chain (they gave 3 false regressions).
STEPS=$(ls -1 "$SYS/steps" 2>/dev/null | sort -V | tr '\n' ' ')
[ -n "$STEPS" ] || { echo "no steps under $SYS/steps"; exit 2; }
echo "line: $STEPS"

ssh -o BatchMode=yes "$REMOTE" "
set -u
module purge 2>/dev/null; module load Apptainer/1.5.0 2>/dev/null
prev=''; seen=''; rc=0
for step in $STEPS; do
    sif=$B/sif/shark-\$step.sif
    [ -s \"\$sif\" ] || { printf '%-28s %s\n' \"\$step\" 'NO IMAGE'; continue; }
    fp=\$(apptainer exec \$sif sh -c 'cat /root/shark/src/protocols/lrs.cpp /root/shark/include/shark/utils/comm.hpp 2>/dev/null | md5sum' 2>/dev/null | awk '{print \$1}')
    mark=''
    if [ -n \"\$prev\" ] && [ \"\$fp\" != \"\$prev\" ]; then
        case \" \$seen \" in
            *\" \$fp \"*) mark='  <-- REGRESSION: an EARLIER step already shipped this library, so a cumulative patch was dropped'; rc=1 ;;
            *) mark='  (library lever introduced here)' ;;
        esac
    fi
    printf '%-28s %s%s\n' \"\$step\" \"\$fp\" \"\$mark\"
    seen=\"\$seen \$fp\"; prev=\$fp
done
exit \$rc
"
rc=$?
[ $rc -eq 0 ] && echo "lineage OK: every step ships its predecessors' library levers"
exit $rc
