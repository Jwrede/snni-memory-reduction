#!/bin/bash
# Copies one step's runs from the cluster into its step directory.
#
#   harvest_step.sh <system-dir> <step_id> <remote> <remote-log-root> [<remote-dir>]
#
# <remote-dir>: cluster run name when it differs from the step id (moai-cpu: p_m<N>_refreeze ->
# m<N>_<lever>); channels looked up under it as well. Absent runs are skipped.
#   <step>        -> logs/            only run whose peaks and wall are published
#   <step>_attrib -> attrib/device/   device composition, allocated bytes
#   <step>_pmrec  -> attrib/resident/ host composition, resident bytes (pmrec + pagemap)
# Transfer: one tarball per run. Large artefacts excluded (EXCL below); CAPTURES keeps sha256 + path.
set -uo pipefail

SYSDIR="${1:?need system dir}"
STEP="${2:?need step id}"
REMOTE="${3:?need remote host}"
LOGROOT="${4:?need remote log root}"
RDIR="${5:-$STEP}"
SD="$SYSDIR/steps/$STEP"

# Absolute remote path required: ~ in the single-quoted remote command never expands, so a tilde
# path silently failed every transfer (six steps "harvested", nothing copied).
case "$LOGROOT" in
    /*) ;;
    *) echo "ERROR: remote log root must be absolute, got '$LOGROOT'" >&2; exit 2 ;;
esac

# Exclusions: measured artefacts that are reproducible INPUTS, not evidence, and large enough to fill
# the volume. Excluded by name (not size, which would drop legitimately-large evidence):
#   heap.dump* : ARION's full Go heap dump, heap.dump.1 = 570,683,505,179 B.
#   spill/*.bank : park banks; MOAI-GPU s6 pulled a live 65.1 GB wpark.bank -> ENOSPC on the next write.
#   pmtable*/devtable* : recorder pointer tables (runtime addresses, dead once the process is gone;
#     derivable from objects_vram.jsonl). devtable joined 2026-08-22 when it became a named file;
#     SIGMA-GPU wrote 31/step at 24 MB -> 5.3 GB over 7 steps.
EXCL="--exclude=*.pickle --exclude=*.bin --exclude=pmtable* --exclude=devtable* --exclude=core.* --exclude=heap.dump*"
EXCL="$EXCL --exclude=spill --exclude=*.bank --exclude=snni_encrypt_spill_* --exclude=snni_param_spill_*"
# Key files a step writes to disk: SIGMA-GPU s6_key_window's 18,075,947,008 B/party, present twice
# (bind mount + container view) -> 64.8 GB harvested, ENOSPC on the next mkdir. Reproducible input.
EXCL="$EXCL --exclude=key_*.dat --exclude=*.key --exclude=models"

have() { ssh "$REMOTE" "[ -d '$LOGROOT/$1' ]"; }

pull() {  # <remote subdir> <local subdir>
    local rd="$1" ld="$2"
    mkdir -p "$ld"
    rm -rf "${ld:?}/"*
    # tar | tar, not a staged file: nothing written twice, and a partial transfer leaves an obviously
    # broken directory rather than a plausible truncated one.
    ssh "$REMOTE" "tar czf - $EXCL -C '$LOGROOT/$rd' ." 2>/dev/null | tar xzf - -C "$ld" || return 1
    ssh "$REMOTE" "cd '$LOGROOT/$rd' && find . \\( -name '*.pickle' -o -name '*.bin' \\) -exec sha256sum {} + 2>/dev/null" \
        | sed "s#\$# ($LOGROOT/$rd)#" > "$ld/CAPTURES" 2>/dev/null
    # A heap dump gets size+path, not a hash: reading 570 GB to checksum costs about as much as
    # copying it, which is what this mechanism avoids.
    ssh "$REMOTE" "cd '$LOGROOT/$rd' && find . -name 'heap.dump*' -printf '%s bytes (no hash: too large to checksum cheaply)  %p\\n' 2>/dev/null" \
        | sed "s#\$# ($LOGROOT/$rd)#" >> "$ld/CAPTURES" 2>/dev/null
    [ -s "$ld/CAPTURES" ] || rm -f "$ld/CAPTURES"
    return 0
}

# Newest RUN_META timestamp under a pulled directory (ISO-8601 sorts as a date). Matches date_utc=
# case-insensitively: it once matched ^DATE_UTC= only, but measure.sh writes it lowercase on 13 of
# 15 systems, so the stale-channel guard below was inert (found via SIGMA-GPU publishing rows whose
# table was 3 days older than their peak).
newest_meta() {  # <local dir> -> the run date, or empty
    grep -hiE '^date_utc=' "$1"/*/RUN_META "$1"/RUN_META 2>/dev/null \
        | cut -d= -f2 | sort | tail -1
}

# A channel older than the clean run it would publish with is refused, not filed: a leftover
# <step>_attrib from a previous arrangement lands in a fresh step and publishes a table from another
# execution (shaft-gpu 2026-08-02: a different MACHINE). Second line of defence after the driver clear.
check_not_stale() {  # <local channel dir> <label>
    local ch="$1" label="$2" c m
    c=$(newest_meta "$SD/logs"); m=$(newest_meta "$ch")
    [ -n "$c" ] && [ -n "$m" ] || return 0
    [ "$m" \< "$c" ] || return 0
    echo "ERROR: the $label channel is OLDER than the clean run it would be published with." >&2
    echo "       clean run  $c" >&2
    echo "       $label     $m" >&2
    echo "Its table describes a different execution. Move it aside on the cluster and re-measure" >&2
    echo "that channel; do not publish a decomposition of a run this step did not make." >&2
    return 1
}

pull "$RDIR" "$SD/logs" || exit 1

# A combined run carries both recorders and produces no separate channels, so it must not be given
# any: leftover 3-run dirs on scratch would otherwise be filed as this step's decomposition. The run
# declares which arrangement it was.
COMBINED=$(grep -h '^combined_channel=1$' "$SD/logs"/*/RUN_META "$SD/logs"/RUN_META 2>/dev/null \
             | wc -l)
if [ "${COMBINED:-0}" -ge 1 ]; then
    echo "combined channel: skipping ${STEP}_attrib and ${STEP}_pmrec (this run carries both recorders)"
    for _old in "$SD/attrib/device" "$SD/attrib/resident"; do
        [ -d "$_old" ] || continue
        mv "$_old" "${_old}.superseded-3run-$(date -u +%Y%m%d)" 2>/dev/null || true
        echo "  moved a previous separate channel aside: ${_old}.superseded-3run-$(date -u +%Y%m%d)"
    done
else
    have "${RDIR}_attrib" && { pull "${RDIR}_attrib" "$SD/attrib/device"   || exit 1
                               check_not_stale "$SD/attrib/device"   "device"   || exit 1; }
    have "${RDIR}_pmrec"  && { pull "${RDIR}_pmrec"  "$SD/attrib/resident" || exit 1
                               check_not_stale "$SD/attrib/resident" "resident" || exit 1; }
fi
# The derivation writes object tables next to the clean run; make_steps.py reads them from the step
# directory. Moving them up here decouples the cluster layout from the published one.
for f in objects_vram.jsonl objects_host.jsonl objects_resident.jsonl; do
    [ -f "$SD/logs/$f" ] && mv -f "$SD/logs/$f" "$SD/$f"
done

echo "harvested $STEP into $SD"
du -sh "$SD"
