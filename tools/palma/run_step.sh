#!/bin/bash
# Submits one step of one system: all replicates and channels, serial, then the derives.
# System settings: palma.conf. Instrument settings: campaign.env.
#
#   run_step.sh <system> <step_id>
#
# Serial: concurrent replicates on a shared filesystem gave walls 229, 252 and 1212 s for one binary.
# Dependency type afterany: a failed replicate stays visible as failed.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SYS="${1:?usage: run_step.sh <system> <step_id>}"
STEP="${2:?usage: run_step.sh <system> <step_id>}"

# Campaign policy first, system config second.
. "$HERE/campaign.env" || exit 1
CONF="$HERE/conf/$SYS.conf"
[ -r "$CONF" ] || { echo "no config for system '$SYS' at $CONF" >&2; exit 1; }

# A conf is sourced, so nothing stops it reintroducing the exact per-system instrument divergence
# this harness exists to remove; the policy is snapshotted before sourcing and any change is a
# hard error.
#
# REPLICATES is the one value a system may lower, with a declared reason (e.g. MOAI-CPU's ~144h
# runs make three replicates a fortnight of machine time). It may only go DOWN, and must be
# declared in that system's MANIFEST.md so the weaker claim travels with every row.
_policy_snapshot() {
    printf '%s|%s|%s|%s|%s\n' \
        "$SNNI_PM_MIN" "$SNNI_PM_FULLSCAN" "$SNNI_PM_FREEZE" "$POLL_MS" "$SNNI_POLICY_VERSION"
}
_before=$(_policy_snapshot)
_reps_before="$REPLICATES"

. "$CONF" || exit 1

if [ "$(_policy_snapshot)" != "$_before" ]; then
    echo "ERROR: $CONF changes a campaign measurement setting." >&2
    echo "  before: $_before" >&2
    echo "  after:  $(_policy_snapshot)" >&2
    echo "Instrument settings are campaign-wide and belong in campaign.env alone. A per-system" >&2
    echo "override is how SHAFT-GPU spent a day at the retired 1 MiB floor with no symptom." >&2
    exit 1
fi
if [ "$REPLICATES" -gt "$_reps_before" ] 2>/dev/null; then
    echo "ERROR: $CONF raises REPLICATES from $_reps_before to $REPLICATES." >&2
    echo "Three is a floor. A system may lower it with a reason in its MANIFEST, never raise it" >&2
    echo "here, or two systems' spreads stop being the same kind of number." >&2
    exit 1
fi
# A caller may raise a lowered count back toward the floor, and only that far: REPLICATES in a
# conf is a system default, but the right count can differ between a BASELINE (the reference
# every later row is judged against) and a STEP (e.g. SIGMA-GPU lowers it for queue exposure on a
# saturated partition). SNNI_REPLICATES may go up toward the campaign floor and never past it, so
# no caller can quietly give any system more replicates than the campaign allows.
if [ -n "${SNNI_REPLICATES:-}" ]; then
    if [ "$SNNI_REPLICATES" -gt "$_reps_before" ] 2>/dev/null; then
        echo "ERROR: SNNI_REPLICATES=$SNNI_REPLICATES is above the campaign floor of $_reps_before." >&2
        echo "Three is a floor, not a default to be exceeded: two systems' spreads have to stay" >&2
        echo "the same kind of number." >&2
        exit 1
    fi
    echo "NOTE: SNNI_REPLICATES=$SNNI_REPLICATES overrides this system's configured $REPLICATES."
    REPLICATES="$SNNI_REPLICATES"
fi
if [ "$REPLICATES" != "$_reps_before" ]; then
    echo "NOTE: $SYS runs $REPLICATES replicate(s), below the campaign floor of $_reps_before."
    echo "      Its MANIFEST.md must declare 'replicates: $REPLICATES' with a reason, and every"
    echo "      published row will carry n_runs=$REPLICATES so the weaker claim travels with it."
fi

: "${WORKDIR:?palma.conf must set WORKDIR}"
: "${SBATCH:?palma.conf must set SBATCH}"
: "${DERIVE:?palma.conf must set DERIVE}"
: "${CHANNELS:?palma.conf must set CHANNELS}"
N="${REPLICATES:?campaign.env must set REPLICATES}"

cd "$WORKDIR" || { echo "no workdir $WORKDIR" >&2; exit 1; }

# Clear the step's result directory: make_steps.py takes a step's peak as the max over every poll
# trace under logs/, so a re-run beside its predecessor would publish the larger of the two under
# one step's name.
#
# Every channel of the step, not just the bare name: clearing only `<step>` leaves a previous
# run's `<step>_attrib`/`<step>_pmrec` in place, so harvest_step.sh (which pulls channels by
# suffix) files an OLD channel's table under the NEW step (measured on shaft-gpu 2026-08-02: a
# device table from a retired machine survived a hardware move this way, dated two days earlier
# with nothing to say so).
STAMP=$(date -u +%Y%m%d-%H%M%S)
for spec in $CHANNELS; do
    sfx="${spec%%:*}"; [ "$sfx" = "-" ] && sfx=""
    d="$WORKDIR/results/$STEP$sfx"
    [ -d "$d" ] || continue
    mv "$d" "$WORKDIR/results/.superseded-$STEP$sfx-$STAMP"
    echo "moved previous results aside: results/.superseded-$STEP$sfx-$STAMP"
done

# The instrument settings that every run of every system carries. Exported by name so that adding
# one here reaches every system at once, which is the entire point of this file existing.
POLICY="SNNI_PM_MIN=$SNNI_PM_MIN,SNNI_PM_FULLSCAN=$SNNI_PM_FULLSCAN"
POLICY="$POLICY,SNNI_PM_FREEZE=$SNNI_PM_FREEZE,POLL_MS=$POLL_MS"
POLICY="$POLICY,SNNI_POLICY_VERSION=$SNNI_POLICY_VERSION"

# SNNI_SBATCH_EXTRA: extra sbatch flags, for PROBES ONLY -- SLURM's own precedence leaves no
# other way, since a `#SBATCH` line overrides the matching `SBATCH_*` env var. Added so a depth
# probe (asks only which stack frame gets recorded, a property of the binary/code path, not the
# node) can run on a free partition instead of queuing behind a baseline that needs a
# bigger-memory node.
#
# A waterfall row must not use this: its peak is compared against its predecessor's under the
# "one system, one machine type" rule, so the step name is checked below rather than trusted.
if [ -n "${SNNI_SBATCH_EXTRA:-}" ]; then
    case "$STEP" in
        p_*|x_*) ;;
        *) echo "ERROR: SNNI_SBATCH_EXTRA is for probes (p_*, x_*), not for '$STEP'." >&2
           echo "A waterfall row measured on other hardware cannot be compared to its parent." >&2
           exit 1 ;;
    esac
    echo "NOTE: extra sbatch flags for this PROBE: $SNNI_SBATCH_EXTRA"
fi

echo "system=$SYS step=$STEP replicates=$N channels='$CHANNELS' policy=$SNNI_POLICY_VERSION"

# SNNI_DEP lets run_line.sh chain one step behind the previous step's last measurement run, so a
# whole line stays serial without this script knowing anything about lines.
prev="${SNNI_DEP:-}"

# And without SNNI_DEP, chain behind whatever this system still has in the queue: two separate
# invocations otherwise know nothing about each other, so submitting a second step while the
# first still ran put two runs of the same system on the partition at once (measured 2026-08-16:
# a probe and a step shared a node, and though different GPUs kept VRAM safe, the step's wall
# time -- the axis that types it free or paid -- absorbed ten minutes of contention with a
# host-heavy recorder). The handle is recorded per system rather than guessed from job names,
# since job names don't reliably match system names (e.g. `moai-gpu` submits jobs called
# `moaigpu`).
LASTJOB="$WORKDIR/.last_measurement_job"
if [ -z "$prev" ] && [ -r "$LASTJOB" ]; then
    _last=$(cat "$LASTJOB" 2>/dev/null)
    # Still in the queue means still PENDING or RUNNING; a finished job leaves squeue entirely.
    if [ -n "$_last" ] && [ -n "$(squeue -h -j "$_last" -o %i 2>/dev/null)" ]; then
        prev="$_last"
        echo "NOTE: $SYS still has job $_last in the queue, so this step is chained behind it."
        echo "      Two runs of one system on the partition at once are not two measurements."
    fi
fi
for r in $(seq 1 "$N"); do
    # CHANNELS: space-separated `suffix:VAR=VAL,...` entries (same spec palma_chain.sh uses). A
    # two-pool system gets one channel per pool: device and host profilers running together once
    # produced a deadlock that cost four revisions of a retired tool.
    for spec in $CHANNELS; do
        suffix="${spec%%:*}"
        chenv="${spec#*:}"
        [ "$suffix" = "-" ] && suffix=""
        name="$STEP$suffix"
        dep=""
        [ -n "$prev" ] && dep="--dependency=afterany:$prev"
        # shellcheck disable=SC2086
        j=$(sbatch --parsable $dep ${SNNI_SBATCH_EXTRA:-} \
            --export="ALL,STEP=$name,RUN=$r,$POLICY${chenv:+,$chenv}" \
            "$SBATCH") || { echo "sbatch failed for $name run$r" >&2; exit 1; }
        echo "  $name run$r job=$j${prev:+  (after $prev)}"
        prev="$j"
    done
done

# Recorded for the NEXT invocation to chain behind: the last MEASUREMENT run, not the last derive,
# since a derive reads finished files and doesn't contend with a run.
echo "$prev" > "$LASTJOB" 2>/dev/null || true

# Derives may run concurrently with each other (CPU work on already-written files, not more
# filesystem writes) but wait for the LAST run so none reads a trace still being written.
for r in $(seq 1 "$N"); do
    j=$(sbatch --parsable --dependency=afterany:"$prev" \
        --export="ALL,STEP=$STEP,RUN=$r,$POLICY" "$DERIVE") \
        || { echo "sbatch failed for derive run$r" >&2; exit 1; }
    echo "  derive run$r job=$j"
done

# The handle a caller chains the NEXT step behind: the last measurement run, not the last derive,
# which wouldn't contend with it anyway.
echo "LAST_JOB=$prev"
