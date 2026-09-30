#!/bin/bash
# Checks that no sbatch carries a default for a campaign measurement setting; exit 1 on violation.
#
#   check_policy.sh [<dir-of-sbatch-files> ...]     default: systems/*/impl (recursively)
#
# Required form: ${VAR:?...}. Forbidden: ${VAR:-default}.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

POLICY_VARS="SNNI_PM_MIN SNNI_PM_FULLSCAN SNNI_PM_FREEZE POLL_MS PM_MIN FULLSCAN REPLICATES"

dirs=("$@")
[ ${#dirs[@]} -eq 0 ] && dirs=(systems)

files=$(find "${dirs[@]}" -type f \( -name '*.sbatch' -o -name 'run_*.sh' \) 2>/dev/null | sort)
[ -n "$files" ] || { echo "no sbatch or runner scripts found under ${dirs[*]}"; exit 0; }

bad=0
for f in $files; do
    for v in $POLICY_VARS; do
        # ${VAR:-default} and VAR=literal hardcode policy; ${VAR:?message} (the accepted form) does
        # not, so require-lines are filtered out after the match (in either quoting).
        hits=$(grep -nE "\\\$\{$v:-|^[[:space:]]*(export[[:space:]]+)?$v=[^\$]" "$f" 2>/dev/null | grep -v ':?' || true)
        if [ -n "$hits" ]; then
            echo "POLICY HARDCODED in $f:"
            printf '%s\n' "$hits" | sed 's/^/    /'
            bad=$((bad + 1))
        fi
    done
done

echo
if [ "$bad" -gt 0 ]; then
    echo "$bad file/variable pair(s) carry their own copy of a campaign measurement setting."
    echo "Take the value from tools/palma/campaign.env and make the script REQUIRE it:"
    echo "    SNNI_PM_MIN=\"\${SNNI_PM_MIN:?campaign.env must supply the recorder floor}\""
    echo "A stale default is silent; a missing variable is loud, and loud is what this needs to be."
    exit 1
fi
echo "OK: no system carries its own copy of a campaign measurement setting"
