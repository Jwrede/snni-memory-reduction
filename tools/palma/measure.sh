# Shared measurement functions for every system's sbatch. Source, do not execute.
#
#   . "$CAMP/harness/measure.sh"
#
# Per-system sbatch keeps: SLURM resources, image, launch, payload process name, gate observable.

# ---------------------------------------------------------------------------------------------
snni_require_policy() {
    # Instrument settings arrive from campaign.env via run_step.sh, required rather than defaulted:
    # a default is silent when forgotten and produces a run that looks correct.
    : "${SNNI_PM_MIN:?campaign.env must supply the recorder floor (run via harness/run_step.sh)}"
    : "${SNNI_PM_FULLSCAN:?campaign.env must supply SNNI_PM_FULLSCAN}"
    : "${SNNI_PM_FREEZE:?campaign.env must supply SNNI_PM_FREEZE}"
    : "${POLL_MS:?campaign.env must supply POLL_MS}"
    : "${SNNI_POLICY_VERSION:?campaign.env must supply SNNI_POLICY_VERSION}"
    # A selected recorder build that is not there would be preloaded as nothing, with only a
    # loader warning: the run would measure without its recorder.
    if [ -n "${SNNI_AR:-}" ] && { [ ! -r "$SNNI_AR/pmrec.so" ] || [ ! -x "$SNNI_AR/pmsample" ]; }; then
        echo "FAILED: SNNI_AR=$SNNI_AR holds no pmrec.so + pmsample" >&2
        exit 2
    fi
}

# ---------------------------------------------------------------------------------------------
snni_prepare_results() {
    # snni_prepare_results <results-dir>
    #
    # Refuses to write into a directory that already holds a run: make_steps.py takes a step's
    # peak as the max over every poll trace under logs/, so a re-run beside its predecessor would
    # publish the larger of the two under one step's name.
    local res="$1"
    if [ -d "$res" ] && [ -n "$(ls -A "$res" 2>/dev/null)" ]; then
        echo "REFUSING: $res already holds a run. Move it aside; a repeat is evidence about" >&2
        echo "reproducibility and must not be silently merged into its predecessor." >&2
        return 3
    fi
    mkdir -p "$res" || return 2
}

# ---------------------------------------------------------------------------------------------
snni_run_meta() {
    # snni_run_meta <results-dir> [extra "key=value" ...]
    #
    # The fields every system records. A published table has to be traceable to the policy it was
    # measured under, which is why the policy version is in here and not only in a log.
    local res="$1"; shift
    {
        echo "step=${STEP:-unset}"
        echo "run=${RUN:-1}"
        echo "date_utc=$(date -u +%FT%TZ)"
        echo "machine=palma:${SLURM_JOB_PARTITION:-unknown} node=$(hostname)"
        echo "slurm_job_id=${SLURM_JOB_ID:-na}"
        echo "policy_version=$SNNI_POLICY_VERSION"
        echo "pm_min=$SNNI_PM_MIN fullscan=$SNNI_PM_FULLSCAN freeze=$SNNI_PM_FREEZE poll_ms=$POLL_MS"
        # This is the REQUEST, labelled as one (2026-08-22): pmrec clamps depth to its own
        # ceiling and used to do so silently while this line wrote the requested number (cost: a
        # depth-7 request recorded as pm_depth=7 but ran at 6). The recorder now announces the
        # EFFECTIVE depth on stderr at init, captured by the run's own log.
        echo "pm_depth_requested=${SNNI_PM_DEPTH:-1}"
        echo "pm_depth_max=${SNNI_PM_DEPTH_MAX:-6}"
        echo "pm_depth=${SNNI_PM_DEPTH:-1}"
        echo "attrib_host=$([ -n "${PMREC:-}" ] && echo pmrec_pagemap || echo off)"
        echo "attrib_device=$([ -n "${ATTRIB:-}" ] && echo torch_state || echo off)"
        snni_meta_optin
        local kv
        for kv in "$@"; do echo "$kv"; done
    } > "$res/RUN_META"
    cat "$res/RUN_META"
}

# ---------------------------------------------------------------------------------------------
snni_meta_optin() {
    # The RUN_META lines of the opt-in instrument settings, each written only when set, so every
    # other run's RUN_META is unchanged. Called by snni_run_meta, and by a sbatch that writes its
    # own RUN_META (SHAFT-GPU).
    # freeze_mode: see snni_start_poller.
    [ -n "${SNNI_PM_FREEZE_MODE:-}" ] && echo "freeze_mode=$SNNI_PM_FREEZE_MODE"
    # A recorder build selected with SNNI_AR (the system's sbatch sets AR from it): which build,
    # and the md5s of the preloaded library and the sampler.
    [ -n "${SNNI_AR:-}" ] && echo "recorder_ar=$SNNI_AR" \
        "pmrec_md5=$(md5sum < "$SNNI_AR/pmrec.so" 2>/dev/null | cut -d' ' -f1)" \
        "pmsample_md5=$(md5sum < "${SNNI_PM_SAMPLE_BIN:-$SNNI_AR/pmsample}" 2>/dev/null | cut -d' ' -f1)"
    return 0
}

# ---------------------------------------------------------------------------------------------
snni_pmrec_inner_env() {
    # snni_pmrec_inner_env <results-dir>
    #
    # Emits the environment the RECORDER needs, printed into the inner shell command rather than
    # passed via `apptainer --env`: apptainer manages LD_PRELOAD itself in some site
    # configurations, silently dropping a preload with no error.
    #
    # `%d` in the table path expands to the pid: multi-process systems (e.g. CrypTen's parties)
    # need one table per pid to keep their objects separable.
    # SNNI_PM_DEPTH decides the table's grouping key: 1 = one allocating function, 2 = the pair
    # (function, caller). Defaulted here (not required) since 1 is what every published table so
    # far was measured at; depth 2 is opt-in per system. Exists because depth 1 sometimes names an
    # unaimable function (e.g. libc10.so/python3.10 addresses). RUN_META records the depth used,
    # since tables at different depths aren't comparable row for row.
    [ -n "${PMREC:-}" ] || return 0
    printf 'export LD_PRELOAD=/ar/pmrec.so SNNI_PM_TABLE=%s/pmtable_%%d SNNI_PM_MIN=%s SNNI_PM_DEPTH=%s;' \
        "$1" "$SNNI_PM_MIN" "${SNNI_PM_DEPTH:-1}"
    # Passed only when set, deliberately: the ceiling defaults to 6, so an unset request produces
    # a byte-identical export line and measurement. See the ceiling's comment in pmrec.c.
    [ -n "${SNNI_PM_DEPTH_MAX:-}" ] && printf ' export SNNI_PM_DEPTH_MAX=%s;' "$SNNI_PM_DEPTH_MAX"
    return 0
}

# ---------------------------------------------------------------------------------------------
snni_descends_from() {
    # snni_descends_from <pid> <ancestor-pid>
    local pid="$1" want="$2" ppid n=0
    while [ "$pid" -gt 1 ] && [ "$n" -lt 40 ]; do
        [ "$pid" = "$want" ] && return 0
        ppid=$(awk '/^PPid:/{print $2}' "/proc/$pid/status" 2>/dev/null) || return 1
        [ -n "$ppid" ] || return 1
        pid="$ppid"; n=$((n + 1))
    done
    return 1
}

snni_find_payload() {
    # snni_find_payload <launcher-pid> <exe-basename> [timeout-tenths]
    # Echoes the pid of the process that HOLDS THE MEMORY, or nothing.
    #
    # Attach to the process that HOLDS THE MEMORY, not the one that looks like the program
    # (measured: cost four separate runs, including a poller that measured a 398 MB launcher while
    # the workers sat at 2.7 GB). Matched on /proc/<pid>/exe, not cmdline: the launcher's
    # descendants (apptainer, inner shell) all carry the payload's name in their cmdline too, so
    # only `exe` says what a process actually IS. `readlink`, not `readlink -f`: the binary's path
    # may not exist on the host, so canonicalising would return empty and filter out every
    # candidate.
    local launcher="$1" exe="$2" tries="${3:-300}" c i
    for i in $(seq 1 "$tries"); do
        for c in $(pgrep -u "$(id -u)" -f "$exe" 2>/dev/null); do
            [ "$c" = "$launcher" ] && continue
            case "$(readlink "/proc/$c/exe" 2>/dev/null)" in
                */"$exe"|*"/$exe (deleted)") ;;
                *) continue ;;
            esac
            if snni_descends_from "$c" "$launcher"; then echo "$c"; return 0; fi
        done
        kill -0 "$launcher" 2>/dev/null || break
        sleep 0.1
    done
    return 1
}

snni_worker_pids_from_files() {
    # snni_worker_pids_from_files <dir> <glob>      e.g. snni_worker_pids_from_files "$RES" 'party*.pid'
    #
    # Prefer this over pattern matching whenever the harness writes its pids: a spawned
    # multiprocessing child's cmdline never contains the script name.
    local dir="$1" glob="$2" f out=""
    for f in "$dir"/$glob; do
        [ -r "$f" ] || continue
        out="$out $(tr -dc '0-9' < "$f")"
    done
    echo "${out# }"
}

# ---------------------------------------------------------------------------------------------
snni_start_poller() {
    # snni_start_poller <results-dir> <outdir-name> <pid> [more pids ...]
    # Echoes the poller pids it started, so the caller can wait on them.
    #
    # One leader freezes every party with one kill: stopping one peer of a multi-party protocol
    # while others run is a protocol fault, not a pause. Followers get SNNI_PM_FREEZE stripped
    # with `env -u` (not just left unset), since --export=ALL would otherwise let a follower thaw
    # its party mid-capture.
    #
    # SNNI_PM_FREEZE_MODE=maxparty (campaign variable, opt-in, 2026-09-29; multi-party only): the
    # leader's poller watches the VmRSS of every party and, whenever the maximum over the parties
    # reaches a new high, stops all parties with the same single kill and captures the party that
    # holds the maximum into that party's own poll_<pid>/peaks. The published peak is the maximum
    # over the parties, so the published party's last capture is frozen at its peak whichever party
    # is heavier. Followers keep their traces and take no captures at new maxima. Unset, the
    # pollers are launched exactly as before.
    local res="$1" outbase="$2"; shift 2
    local all="$*" first="$1" p pids=""
    local lead=1 mp=""
    if [ "${SNNI_PM_FREEZE_MODE:-}" = maxparty ] && [ "$(echo "$all" | wc -w)" -gt 1 ]; then
        mp=1
        echo "freeze mode maxparty: the leader freezes [$all] at new highs of the maximum over the parties and captures the party holding it" \
            >> "$res/poller_attach.log"
    elif [ -n "${SNNI_PM_FREEZE_MODE:-}" ]; then
        echo "freeze mode '${SNNI_PM_FREEZE_MODE}' not applied to party set [$all]" >> "$res/poller_attach.log"
    fi
    for p in $all; do
        local out="$res/$outbase"
        [ "$(echo "$all" | wc -w)" -gt 1 ] && out="$res/${outbase}_$p"
        # SNNI_PM_SAMPLE_BIN names another sampler binary (2026-09-18: pmsample_chain, which keys
        # rows on the whole frame chain); the default stays so runs already in flight keep theirs.
        if [ -n "$mp" ] && [ "$lead" = 1 ]; then
            SNNI_PM_MAXPARTY_ROLE=leader \
            SNNI_PM_MAXPARTY_OUT="$res/${outbase}_%d" \
            SNNI_PM_MAXPARTY_TABLE="${PMREC:+$res/pmtable_%d}" \
            SNNI_PM_TABLE="${PMREC:+$res/pmtable_$p}" \
            SNNI_PM_SAMPLE="${PMREC:+${SNNI_PM_SAMPLE_BIN:-$AR/pmsample}}" \
            SNNI_PM_FULLSCAN="$SNNI_PM_FULLSCAN" \
            SNNI_PM_FREEZE="$SNNI_PM_FREEZE" \
            SNNI_PM_FREEZE_PIDS="$all" \
            SNNI_POOLREC_TABLE="${SNNI_POOLREC:+$res/pooltable_$p}" \
            SNNI_MARKER_FILE="${SNNI_MARKER_SAMPLE:+$res/markers.txt}" \
                bash "$CAMP/poll_peak.sh" "$p" "$POLL_MS" "$out" \
                    > "$res/poller_$p.log" 2>&1 &
        elif [ -n "$mp" ]; then
            env -u SNNI_PM_FREEZE -u SNNI_PM_FREEZE_PIDS \
            SNNI_PM_MAXPARTY_ROLE=follower \
            SNNI_PM_TABLE="${PMREC:+$res/pmtable_$p}" \
            SNNI_PM_SAMPLE="${PMREC:+${SNNI_PM_SAMPLE_BIN:-$AR/pmsample}}" \
            SNNI_PM_FULLSCAN="$SNNI_PM_FULLSCAN" \
            SNNI_POOLREC_TABLE="${SNNI_POOLREC:+$res/pooltable_$p}" \
            SNNI_MARKER_FILE="${SNNI_MARKER_SAMPLE:+$res/markers.txt}" \
                bash "$CAMP/poll_peak.sh" "$p" "$POLL_MS" "$out" \
                    > "$res/poller_$p.log" 2>&1 &
        elif [ "$lead" = 1 ]; then
            SNNI_PM_TABLE="${PMREC:+$res/pmtable_$p}" \
            SNNI_PM_SAMPLE="${PMREC:+${SNNI_PM_SAMPLE_BIN:-$AR/pmsample}}" \
            SNNI_PM_FULLSCAN="$SNNI_PM_FULLSCAN" \
            SNNI_PM_FREEZE="$SNNI_PM_FREEZE" \
            SNNI_PM_FREEZE_PIDS="$all" \
            SNNI_POOLREC_TABLE="${SNNI_POOLREC:+$res/pooltable_$p}" \
            SNNI_MARKER_FILE="${SNNI_MARKER_SAMPLE:+$res/markers.txt}" \
                bash "$CAMP/poll_peak.sh" "$p" "$POLL_MS" "$out" \
                    > "$res/poller_$p.log" 2>&1 &
        else
            env -u SNNI_PM_FREEZE -u SNNI_PM_FREEZE_PIDS \
            SNNI_PM_TABLE="${PMREC:+$res/pmtable_$p}" \
            SNNI_PM_SAMPLE="${PMREC:+${SNNI_PM_SAMPLE_BIN:-$AR/pmsample}}" \
            SNNI_PM_FULLSCAN="$SNNI_PM_FULLSCAN" \
            SNNI_POOLREC_TABLE="${SNNI_POOLREC:+$res/pooltable_$p}" \
            SNNI_MARKER_FILE="${SNNI_MARKER_SAMPLE:+$res/markers.txt}" \
                bash "$CAMP/poll_peak.sh" "$p" "$POLL_MS" "$out" \
                    > "$res/poller_$p.log" 2>&1 &
        fi
        pids="$pids $!"
        echo "poller on pid $p (leader=$lead) -> $out" >> "$res/poller_attach.log"
        lead=0
    done
    echo "${pids# }"
    [ -n "$first" ] || return 1
}

# ---------------------------------------------------------------------------------------------
snni_finish() {
    # snni_finish <results-dir> <t0-epoch> <rc>
    #
    # wall_seconds.txt is what runtime_delta_pct is built from. The freeze total lands in the poll
    # trace and make_steps.py subtracts it, so what is written here is the raw wall.
    local res="$1" t0="$2" rc="$3"
    echo $(( $(date +%s) - t0 )) > "$res/wall_seconds.txt"
    local peak
    peak=$(cat "$res"/*/vmrss.log 2>/dev/null \
        | awk 'NR>1 && $2 ~ /^[0-9]+$/{if($2>m)m=$2}END{print m+0}')
    {
        echo "rc=$rc"
        echo "wall_s=$(cat "$res/wall_seconds.txt")"
        echo "polled_peak_kb=$peak"
    } > "$res/PEAK.txt"
    cat "$res/PEAK.txt"
    return "$rc"
}
