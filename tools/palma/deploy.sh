#!/bin/bash
# Copies the shared harness to a system's cluster working directory.
#
#   deploy.sh <system> [remote]        remote defaults to `palma`
#
# Copies: campaign.env, the drivers, the system's conf. Not copied: the system's own sbatch.
# Transfer by scp. Refuses while this system has queued jobs (`squeue -o %Z`, scoped to its dir).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SYS="${1:?usage: deploy.sh <system> [remote]}"
REMOTE="${2:-palma}"

CONF="$HERE/conf/$SYS.conf"
[ -r "$CONF" ] || { echo "no config for system '$SYS' at $CONF" >&2; exit 1; }
# shellcheck disable=SC1090
. "$CONF" || exit 1
: "${WORKDIR:?conf must set WORKDIR}"

n=$(ssh -o BatchMode=yes "$REMOTE" \
      "squeue -u \$USER -h -o %Z 2>/dev/null | grep -c '^$WORKDIR' || true" 2>/dev/null)
if [ -z "$n" ]; then
    echo "could not reach $REMOTE to check the queue; not deploying blind" >&2
    exit 1
fi
if [ "$n" -gt 0 ] && [ -z "${SNNI_DEPLOY_ANYWAY:-}" ]; then
    echo "$n job(s) for $SYS still queued or running in $WORKDIR." >&2
    echo "Deploy between lines, not during one: otherwise the queue and the scripts on disk" >&2
    echo "stop describing the same thing. Override with SNNI_DEPLOY_ANYWAY=1 if you are sure." >&2
    exit 1
fi

# ONE copy of the shared harness, at the campaign root, not one per system: a per-system copy let
# SHAFT-GPU sit at a retired recorder floor for a day after another system's was corrected.
CAMPROOT=$(dirname "$WORKDIR")
case "$WORKDIR" in
    */snni_campaign/*) CAMPROOT=${WORKDIR%/snni_campaign/*}/snni_campaign ;;
    *)                 CAMPROOT=$(dirname "$WORKDIR")/snni_campaign ;;
esac

echo "deploying shared harness to $REMOTE:$CAMPROOT/harness"
ssh -o BatchMode=yes "$REMOTE" "mkdir -p '$CAMPROOT/harness/conf'" || exit 1
scp -q "$HERE/campaign.env" "$HERE/measure.sh" "$HERE/run_step.sh" "$HERE/run_line.sh" \
    "$REMOTE:$CAMPROOT/harness/" || exit 1
scp -q "$HERE/conf/"*.conf "$REMOTE:$CAMPROOT/harness/conf/" || exit 1
ssh -o BatchMode=yes "$REMOTE" "chmod +x '$CAMPROOT/harness/'run_*.sh" || exit 1

# THE ATTRIBUTION PRODUCERS TRAVEL WITH THE HARNESS: a tool deployed by hand once and never again
# silently stops existing on the cluster (measured: a missing attrib_go file failed ARION's census
# after a 2-hour run and a 570 GB heap dump, with only "TRIGGER"/"USR1 sent" logged).
# THE SHARED TOOLS THEMSELVES, for the same reason: a file that exists in the repository and
# differs on the machine is worse than one that is missing (measured: a bug fix sat uncorrected on
# the cluster for ten days because the repo and cluster copies silently diverged).
for _t in derive_host_objects.sh symbolise.py smaps_objects.py poll_peak.sh torch_peak_objects.py; do
    [ -r "$HERE/../$_t" ] || continue
    scp -q "$HERE/../$_t" "$REMOTE:$CAMPROOT/$_t" || exit 1
done
echo "  shared tools: $(ls "$HERE/.."/derive_host_objects.sh "$HERE/.."/symbolise.py 2>/dev/null | wc -l)+ file(s)"

for _at in "$HERE/../attrib_"*; do
    [ -d "$_at" ] || continue
    _n=$(basename "$_at")
    ssh -o BatchMode=yes "$REMOTE" "mkdir -p '$CAMPROOT/$_n'" || exit 1
    # Only sources; a built recorder (.so) is architecture-specific, built on the cluster. .cu and
    # .sbatch travel too (since 2026-08-08): a fix nobody can test on the machine that runs it is
    # not a fix, and the recorder's selftest is what proves a change to it.
    _files=$(ls "$_at"/*.py "$_at"/*.c "$_at"/*.cu "$_at"/*.sbatch "$_at"/*.sh 2>/dev/null)
    [ -n "$_files" ] && { scp -q $_files "$REMOTE:$CAMPROOT/$_n/" || exit 1; }
    echo "  $_n: $(echo "$_files" | wc -w) file(s)"
done

# The system's own scripts, which are legitimately per-system and live beside the code they launch.
IMPL="$HERE/../../systems/$SYS/impl"
if [ -d "$IMPL" ]; then
    # `*.def` joined the list on 2026-08-23: an apptainer definition IS the measured program (pins
    # the upstream commit, applies the step's patch, carries marker checks), and it used to reach
    # the cluster only by hand -- found when a new definition was deployed and simply wasn't there.
    files=$(find "$IMPL" -maxdepth 1 -type f \( -name '*.sbatch' -o -name '*.sh' -o -name '*.def' \) 2>/dev/null)
    if [ -n "$files" ]; then
        echo "deploying $SYS scripts to $REMOTE:$WORKDIR"
        # shellcheck disable=SC2086
        scp -q $files "$REMOTE:$WORKDIR/" || exit 1
        ssh -o BatchMode=yes "$REMOTE" "chmod +x '$WORKDIR/'*.sh 2>/dev/null; true"
    fi

    # THE MEASURED PROGRAM AND ITS LEVERS ARE PYTHON, AND THEY WERE NEVER DEPLOYED before this: the
    # find above matches only `*.sbatch`/`*.sh`, so driver scripts and monkeypatch levers reached
    # the cluster once by hand and every later edit stayed on the laptop (cost two runs: one
    # refused loudly on a missing bind source, one would have raised ModuleNotFoundError
    # in-container). Deploying the files removes the question.
    _pys=$(find "$IMPL" -maxdepth 1 -type f -name '*.py' 2>/dev/null)
    if [ -n "$_pys" ]; then
        echo "deploying $SYS python ($(echo "$_pys" | wc -l) file(s))"
        # shellcheck disable=SC2086
        scp -q $_pys "$REMOTE:$WORKDIR/" || exit 1
    fi
    # Source subdirectories, measured rather than guessed: these are the names the thirteen
    # systems actually use next to `impl/` (a prior single-name list cost a queue slot when a
    # cluster copy of `patches/` went stale). `__pycache__` excluded: a laptop-interpreter build
    # product that can get a stale .pyc imported in a container with a different Python.
    for _sub in levers patches borrowed src overlay base_patch; do
        [ -d "$IMPL/$_sub" ] || continue
        _f=$(find "$IMPL/$_sub" -maxdepth 1 -type f ! -name '*.pyc' 2>/dev/null)
        if [ -n "$_f" ]; then
            echo "deploying $SYS $_sub ($(echo "$_f" | wc -l) file(s))"
            ssh -o BatchMode=yes "$REMOTE" "mkdir -p '$WORKDIR/$_sub'" || exit 1
            # shellcheck disable=SC2086
            scp -q $_f "$REMOTE:$WORKDIR/$_sub/" || exit 1
        fi
    done

    # STEP DECLARATIONS, for env-gated systems (17.08.): never deployed before, since the find
    # above is -maxdepth 1 and matches only scripts. A new step's declaration then stayed on the
    # laptop while the job ran -- one system refused loudly, another silently measured the
    # baseline under the new step's name.
    if [ -d "$IMPL/stepenv" ]; then
        envs=$(find "$IMPL/stepenv" -maxdepth 1 -type f -name '*.env' 2>/dev/null)
        if [ -n "$envs" ]; then
            echo "deploying $SYS step declarations ($(echo "$envs" | wc -l) file(s))"
            ssh -o BatchMode=yes "$REMOTE" "mkdir -p '$WORKDIR/stepenv'" || exit 1
            # shellcheck disable=SC2086
            scp -q $envs "$REMOTE:$WORKDIR/stepenv/" || exit 1
        fi
    fi
fi

echo "deployed. Run a step with:"
echo "  ssh $REMOTE '$CAMPROOT/harness/run_step.sh $SYS <step_id>'"
