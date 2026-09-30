#!/bin/bash
# Submits one step (clean run, then attribution runs) as one SLURM dependency chain.
#
#   palma_chain.sh <sbatch> <step> [attrib-spec ...]
#   e.g. palma_chain.sh .../shaft_gpu_campaign.sbatch s1_limb_loop \
#                       _attrib:ATTRIB=1 _pmrec:PMREC=1  SHAFT_LIMB_LOOP=1
#
# `VAR=VAL` (no leading `_`): lever, passed to every run. `_suffix:VAR=VAL,...`: attribution run.
# Attribution runs depend on the clean run with afterok, not on each other.
set -uo pipefail

SB="${1:?need sbatch path}"
STEP="${2:?need step}"
shift 2

REMOTE="${SNNI_REMOTE:-palma}"
LEVERS=""
SPECS=()
for a in "$@"; do
    case "$a" in
        _*:*) SPECS+=("$a") ;;
        *=*)  LEVERS="$LEVERS,$a" ;;
        *)    echo "ERROR: cannot tell if '$a' is a lever or an attribution spec" >&2; exit 2 ;;
    esac
done

sub() {  # <name> <extra-export> [dependency]
    local name="$1" extra="$2" dep="${3:-}"
    local cmd="sbatch --parsable --job-name='${name:0:14}'"
    [ -n "$dep" ] && cmd="$cmd --dependency=afterok:$dep"
    cmd="$cmd --export='ALL,STEP=$name$LEVERS$extra' '$SB'"
    ssh "$REMOTE" "$cmd" 2>&1
}

CLEAN=$(sub "$STEP" "")
case "$CLEAN" in ''|*[!0-9]*) echo "ERROR: clean submit failed: $CLEAN" >&2; exit 2 ;; esac
echo "$STEP clean=$CLEAN"

for spec in "${SPECS[@]}"; do
    suffix="${spec%%:*}"
    vars="${spec#*:}"
    id=$(sub "$STEP$suffix" ",${vars//,/,}" "$CLEAN")
    case "$id" in ''|*[!0-9]*) echo "WARNING: $STEP$suffix submit failed: $id" >&2; continue ;; esac
    echo "$STEP$suffix=$id (afterok:$CLEAN)"
done
