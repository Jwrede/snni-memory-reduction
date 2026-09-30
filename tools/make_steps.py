#!/usr/bin/env python3
"""Generates steps.csv and per-step peak-objects.md from raw step logs (`--check`: regenerate and compare).

Layout:

    systems/<system>/
      MANIFEST.md                 per-pool targets, see read_targets()
      steps/<step_id>/
        levers.env                the lever settings for this step
        logs/                     the CLEAN run: peaks and wall come from here only
        attrib/                   the attribution run: composition only, never peaks
        objects_vram.jsonl        canonical device decomposition (written by the attrib tools)
        objects_host.jsonl        canonical host decomposition

Pools are reported separately, never summed. Attacked pool: larger W = (peak - overhead) / target,
overhead = the pool's library term (read_manifest()).
"""
import csv
import glob
import io
import json
import os
import re
import statistics
import sys

# How far below its own peak a decomposition may be sampled before it stops describing that peak:
# warned from 1%, refused from here (a table this far below pairs with the peak's smaps to produce a
# residual larger than the peak, as a LLAMA dealer run did at 900808 kB vs its clean run's 1465928).
MISALIGN_FAIL_PCT = 10.0


SCHEMA = [
    "step_id", "lever", "impl_fingerprint", "pool_attacked", "pool_required",
    "peak_vram", "delta_vram", "overhead_vram", "w_vram",
    "peak_host", "delta_host", "overhead_host", "w_host",
    # Replication: peak_host is the MEDIAN of n_runs; extremes published (3 points do not support a
    # summary stat). The spread makes "free" checkable: a delta smaller than it is the noise floor.
    "n_runs", "peak_host_min", "peak_host_max", "spread_host_pct",
    # Multi-party: peak_host is the maximum over the parties of each run (median over runs of that
    # per-run maximum); peak_host_other is the largest OTHER party's raw peak of the same median run.
    # peak_party names whose peak was published (leader / non-leader, the freeze leader as named in
    # poller_attach.log); decomposed_party names whose object table the row's decomposition is.
    # They differ only where the heavier party has no table of its own. Empty on single-party systems.
    "peak_host_other", "peak_party", "decomposed_party",
    "cross_pool",
    "object_targeted", "object_size", "object_frac_of_attributed", "object_source",
    "unattributed_vram", "unattributed_host",
    # unattributed = 1 - total/peak (tends to 0, residual absorbs the rest); named_frac = share of
    # peak with a real identity (heap:/file:) vs located-but-unnamed anon:. See PROCEDURE Coverage.
    "named_frac_vram", "named_frac_host",
    # sum_heap: named-heap total = lower bound on what a step could target. peak_anon: anon: total =
    # upper bound on what might still hide there. The two brackets replaced the retired W_live.
    "peak_anon", "sum_heap",
    # Published, not enforced: targeted object size vs anon: total. Below 1.0 the peak has stopped
    # being made of nameable objects (a system finding, not a step defect). See PROCEDURE Coverage.
    "object_vs_unnamed",
    # Runtime axis. cost_type (free/paid/"") compares runtime_delta_pct against this step's OWN wall
    # spread, never a fixed threshold. (peak_location column removed 2026-08-10: nothing wrote it.)
    "wall_s", "frozen_ms", "wall_spread_pct", "runtime_delta_pct", "cost_type",
    # Transcript: phase-B evidence that a step did the same protocol work when the gate cannot tell
    # a repartition (different masks, moved output) from a real change. Retracted BumbleBee x7 (+41% bytes).
    "comm_bytes", "comm_rounds",
    "correctness_gate",
    "conformant", "object_attacked", "object_required", "object_conformant", "evidence",
]


def read_manifest(sysdir):
    """Per-pool targets and library-overhead terms from MANIFEST.md (both needed for W).

    Expected lines: target_{vram,host}_kb, overhead_{vram,host}_kb. Overhead is excluded from W
    (loading cost, not protocol memory); `measured_per_run` reads it from the step's own smaps.
    """
    path = os.path.join(sysdir, "MANIFEST.md")
    t = {"vram": None, "host": None}
    o = {"vram": 0, "host": 0}
    if os.path.exists(path):
        txt = open(path, errors="replace").read()
        # Every key is anchored to line start, so a retired value mentioned in prose does not
        # re-declare it (ARION's manifest tripped the unanchored search into reading two tolerances).
        def _key(pattern):
            return re.search(rf"^\s*{pattern}", txt, re.MULTILINE)
        for pool in ("vram", "host"):
            m = _key(rf"target_{pool}_kb:\s*([0-9]+)")
            if m:
                t[pool] = int(m.group(1))
            m = _key(rf"overhead_{pool}_kb:\s*(measured_per_run|[0-9]+)")
            if m:
                o[pool] = None if m.group(1) == "measured_per_run" else int(m.group(1))
        # At most one tolerance key. gate_tolerance is relative (error scales with value);
        # gate_tolerance_abs is absolute (fixed-point ULP, or observables of structural zeros).
        for key in ("gate_tolerance", "gate_tolerance_abs"):
            m2 = _key(rf"{key}:\s*([0-9.eE+-]+)")
            if m2:
                try:
                    t[key] = float(m2.group(1))
                except ValueError:
                    pass
        m = _key(r"gate_tier:\s*(T[0-3]s?)")
        if m:
            t["gate_tier"] = m.group(1)
        # A single-process, non-interactive system exchanges no bytes, so phase B's "same
        # transcript" condition has nothing to compare; the manifest must SAY so (with the reason)
        # for the condition to be read as vacuous rather than as missing evidence.
        m = _key(r"transcript_none:\s*(.+?)\s*$")
        if m:
            t["transcript_none"] = m.group(1)
        # A one-replicate line may declare the run-to-run spread it measured elsewhere (a re-measured
        # baseline, a probe pair), so that the inert-step guard still applies at n=1. Without it a
        # -0.2% step passed as a reduction (MOAI-CPU m6_att_output_release, 2026-09-21).
        m = _key(r"spread_host_pct_declared:\s*([0-9.]+)")
        if m:
            t["spread_host_pct_declared"] = float(m.group(1))
        # Three replicates is the floor, lowerable only by an explicit declaration with a reason
        # (MOAI-CPU is ~144 h per run).
        m = _key(r"replicates:\s*([0-9]+)")
        if m:
            t["replicates"] = int(m.group(1))
    return t, o


# Shapes identifying a mapping as loaded code, not program data.
_LIB_SHAPE = re.compile(r"\.so(\.[0-9]+)*$|\.dylib$")
# CUDA driver device nodes, mapped into every GPU process. Platform, not program.
_DEV_SHAPE = re.compile(r"^(nvidia|dev/|zero$|null$|urandom$)")
# Measured executables, named explicitly (a binary has no suffix to recognise); an unclassified
# file: mapping is reported below rather than guessed.
_EXEC_NAMES = {"python3.10", "python3.9", "python3",          # interpreters (SHAFT, PUMA, LLAMA-D)
               "test",                                        # MOAI-GPU
               "sigma",                                       # SIGMA-GPU
               "op_bench",                                    # BumbleBee
               "BOLT_BERT",                                   # BOLT
               "bertbenchmark",                               # LLAMA online
               "benchmark-bert_instrumented",                 # SHARK, SHARK-Dealer
               "arion",                                       # ARION (Go)
               "main", "a.out"}
# Small system files (locale/gconv/timezone), grouped by name so no size threshold is invented.
_SYS_NAMES = {"LC_CTYPE", "LC_MESSAGES", "LC_TIME", "LC_NUMERIC", "LC_COLLATE", "locale-archive",
              "localtime", "(deleted)", "gconv-modules.cache"}


def is_loaded_code(label):
    """True if this `file:` mapping is loaded code or a platform mapping rather than program data."""
    b = label[len("file:"):] if label.startswith("file:") else label
    return bool(_LIB_SHAPE.search(b)) or bool(_DEV_SHAPE.match(b)) \
        or b in _EXEC_NAMES or b in _SYS_NAMES


def measured_overhead(objs):
    """The library-overhead term measured from this step's own decomposition.

    THE PREMISE THIS USED TO REST ON WAS "`file:` entries are file-backed mappings, i.e. exactly the
    loaded libraries", and it held only as long as the only files a measured program mapped were its
    libraries. **SIGMA-GPU's `s6_dealer_window` broke it by design**: that step makes the program
    write its key material to disk and map the file back, so 817 MiB of protocol keys arrived as
    `file:key_0.dat` and were subtracted from `W` as if they were library text. `W` fell 3.47 -> 2.02
    on a step whose peak moved 0.25%.

    **The defect flatters exactly the levers this campaign builds.** A step that spills data to disk
    and maps it would be paid twice: once for the residency it genuinely removes, and once because
    whatever stays resident is reclassified as somebody else's code. It was not new either -- SHAFT-GPU
    carried a 433.3 MB hash-named model blob in all five of its rows, annotated in its tables as
    "mapped, excluded from W by policy". The annotation was honest; the policy was wrong.

    So overhead is now LOADED CODE ONLY: shared objects, the mapped executable, device nodes and the
    small system tables every process carries. A file the program itself opened stays in the peak,
    where it is rankable and, like any `file:` row, not targetable.

    The recorder's own table is file-backed too and is labelled `instrument:` by smaps_objects.py
    precisely so it does NOT land here -- it is taken off the peak instead, and counting it in both
    places would subtract the instrument twice.
    """
    return sum(v for k, v in objs.items()
               if k.startswith("file:") and is_loaded_code(k)) // 1024


def instrument_kb(step_dir):
    """Resident bytes contributed by the recorder itself, subtracted from the peak exactly.

    The recorder adds only its file-backed table (a named mapping, read from the peak snapshot at
    the peak instant, not a calibration constant), which is why one run per step suffices. Sums
    EVERY instrument: row of the step's own host table, so a second (device) recorder's devtable_ is
    subtracted too; INSTRUMENT_KB is the fallback when the object table is missing.
    """
    from_objs = 0
    for run_dir in sorted(glob.glob(os.path.join(step_dir, "logs", "*"))) + [step_dir]:
        objs, _tot, _rec = load_objects(step_dir, "host", run_dir if run_dir != step_dir else None)
        if not objs:
            continue
        b = sum(v for k, v in objs.items()
                if isinstance(k, str) and k.startswith("instrument:")
                and isinstance(v, (int, float)))
        from_objs = max(from_objs, int(b // 1024))
    if from_objs:
        return from_objs
    tot = 0
    for f in glob.glob(os.path.join(step_dir, "logs/**/peaks/INSTRUMENT_KB"), recursive=True):
        try:
            tot = max(tot, int(open(f, errors="replace").read().strip()))
        except (ValueError, OSError):
            pass
    return tot


def poll_peak(step_dir):
    """Host peak and how obtained: (peak_kb, polled_max_kb, source).

    peak VALUE is VmHWM (kernel high-water, catches sub-interval spikes) when the trace carries it,
    else the 100 ms polled max; polled_max is the instant the snapshot/table were taken. A
    disagreement means the peak was a spike the poller missed (measured: LLAMA d2, 1465928 vs 900808 kB).
    """
    reps = host_replicates(step_dir)
    if not reps:
        return 0, 0, "polled"
    # MEDIAN replicate, not the max: names an actual run so peak and object table share one instant.
    mid = sorted(reps, key=lambda r: r["peak"])[len(reps) // 2]
    return mid["peak"], mid["polled"], mid["source"]


def _trace_peak(path):
    """(peak_kb, polled_max_kb, source) for ONE poll trace."""
    peak, hwm = 0, 0
    with open(path, errors="replace") as fh:
        for ln in fh:
            if ln.startswith("#"):
                m = re.search(r"hwm_kb=([0-9]+)", ln)
                if m:
                    hwm = max(hwm, int(m.group(1)))
                continue
            p = ln.split()
            if len(p) >= 2 and p[1].isdigit():
                peak = max(peak, int(p[1]))
            if len(p) >= 4 and p[3].isdigit():
                hwm = max(hwm, int(p[3]))
    return (hwm or peak), peak, ("VmHWM" if hwm else "polled")


def _run_rejected(trace_dir):
    """True if this run's own gate observable is missing or truncated.

    Walks up from the poll trace to the run directory and looks for the harness's PEAK.txt. A run
    whose gate output is absent or whose declared sizes are zero did not complete the protocol,
    whatever its exit code said, and must not contribute a replicate.
    """
    d = trace_dir
    for _ in range(3):
        pk = os.path.join(d, "PEAK.txt")
        if os.path.exists(pk):
            txt = open(pk, errors="replace").read()
            sizes = re.findall(r"_dat_B=([0-9]+)", txt)
            if sizes and all(int(s) == 0 for s in sizes):
                return True
            km = os.path.join(d, "keys.md5")
            # Present but empty means the gate observable was never produced.
            if os.path.exists(km) and os.path.getsize(km) == 0:
                return True
            return False
        d = os.path.dirname(d)
    # No PEAK.txt above the trace: harness writes it at run END, so its absence means in-flight or
    # died mid-flight (an ordinary-looking trace). Not a replicate. (All 93 published runs carry it.)
    return True


def _leader_traces(run_dir):
    """Map poll-dir basename -> leader flag, from the run's poller_attach.log.

    Only the freeze leader's captures are atomic (it holds the whole party set stopped); followers
    sample without freezing. Each attached poller logs "... (leader=<0|1>) -> <poll dir>".
    """
    out = {}
    p = os.path.join(run_dir, "poller_attach.log")
    if os.path.exists(p):
        for ln in open(p, errors="replace"):
            m = re.search(r"\(leader=([01])\)\s*->\s*(\S+)", ln)
            if m:
                out[os.path.basename(m.group(2).rstrip("/"))] = m.group(1) == "1"
    return out


def _trace_frozen_ms(path):
    """The `# frozen_ms=` footer of one poll trace, 0 if absent."""
    v = 0
    with open(path, errors="replace") as fh:
        for ln in fh:
            if ln.startswith("#"):
                m = re.search(r"frozen_ms=([0-9]+)", ln)
                if m:
                    v = max(v, int(m.group(1)))
    return v


def host_replicates(step_dir):
    """One entry per independent run (a logs/ dir with its own poll trace; multi-party = one entry).

    A run's peak is the MAXIMUM over its parties (decision of 2026-09-29): every party has to fit on
    its machine, so the heavier one sets the memory requirement. Until then the freeze leader's peak
    was published, because only the leader is captured atomically. The leader is still identified,
    so that the row can name whose peak it publishes: from poller_attach.log, else the one trace
    with a nonzero frozen_ms (leader_known=False when neither decides). Ties go to the leader.

    Returns [{dir, peak, polled, source, other_peaks, leader_known, party, pid, traces}], sorted by
    peak; party is "leader" / "non-leader" ("" for a single trace), traces is {pid: raw peak}.
    """
    byrun = {}
    for pat in ("logs/**/vmrss.log", "logs/**/party*_vmrss.log"):
        for f in glob.glob(os.path.join(step_dir, pat), recursive=True):
            peak, polled, src = _trace_peak(f)
            if not peak:
                continue
            # A harness-rejected run leaves an ordinary-looking trace; excluded so it does not enter
            # the median/spread (e.g. d6_ars_free's starved replicate that exited 5 on the size check).
            if _run_rejected(os.path.dirname(f)):
                continue
            # Run dir = trace's grandparent when it sits in a poll_* output dir, else its parent.
            d = os.path.dirname(f)
            run = os.path.dirname(d) if os.path.basename(d).startswith("poll") else d
            byrun.setdefault(run, []).append(
                {"trace_dir": os.path.basename(d), "file": f,
                 "peak": peak, "polled": polled, "source": src})
    out = []
    for run, traces in byrun.items():
        leader, known = None, len(traces) == 1
        if len(traces) > 1:
            leaders = _leader_traces(run)
            led = [t for t in traces if leaders.get(t["trace_dir"])]
            if len(led) != 1:
                led = [t for t in traces if _trace_frozen_ms(t["file"]) > 0]
            if len(led) == 1:
                leader, known = led[0], True
        chosen = max(traces, key=lambda t: (t["peak"], t is leader))
        party = "" if len(traces) == 1 else (
            "leader" if chosen is leader else ("non-leader" if known else "unknown"))
        others = sorted((t["peak"] for t in traces if t is not chosen), reverse=True)
        out.append({"dir": run, "peak": chosen["peak"], "polled": chosen["polled"],
                    "source": chosen["source"], "other_peaks": others, "leader_known": known,
                    "party": party, "pid": _trace_pid(chosen["trace_dir"]),
                    "leader_pid": _trace_pid(leader["trace_dir"]) if leader else None,
                    "traces": {_trace_pid(t["trace_dir"]): t["peak"] for t in traces}})
    return sorted(out, key=lambda r: r["peak"])


def _trace_pid(trace_dir):
    """The pid in a per-party poll directory name (`poll_<pid>`), or None for a single `poll`."""
    m = re.match(r"poll_(\d+)$", trace_dir)
    return m.group(1) if m else None


def _table_pid(path):
    """The pid whose host object table this file is: from `objects_host_<pid>.jsonl`, else from the
    recorder's own row `instrument:pmtable_<pid>` inside it (SHARK names its tables by party)."""
    m = re.search(r"objects_host_(\d+)\.jsonl$", path)
    if m:
        return m.group(1)
    with open(path, errors="replace") as fh:
        m = re.search(r'"instrument:pmtable_(\d+)"', fh.read())
    return m.group(1) if m else None


def _maxparty_capture(run_dir, pid):
    """The `peaks/CAPTURE` record of party `pid` as a dict, when the run was measured with
    freeze_mode=maxparty (RUN_META), else None. Written by poll_peak.sh's snapshot_maxparty."""
    meta = os.path.join(run_dir, "RUN_META")
    if not pid or not os.path.exists(meta) or not re.search(
            r"^freeze_mode=maxparty$", open(meta, errors="replace").read(), re.M):
        return None
    p = os.path.join(run_dir, f"poll_{pid}", "peaks", "CAPTURE")
    if not os.path.exists(p):
        return {}
    return dict(t.split("=", 1) for t in open(p, errors="replace").read().split() if "=" in t)


def _table_gap_pct(path, peak_kb):
    """How far below `peak_kb` (raw) the table's own instant lies, in percent, or None."""
    if not peak_kb:
        return None
    _o, _t, meta = load_objects(None, "host", None, path)
    inst = meta.get("sample_rss_kb") or meta.get("snapshot_rss_kb")
    return 100.0 * (peak_kb - inst) / peak_kb if inst else None


def host_table(rep):
    """(path, pid) of the host object table that describes this replicate's published party.

    The published party's own table when the run carries one (per-party tables: SHARK, BumbleBee,
    BOLT, PUMA), else the run's `objects_host.jsonl`, which on the other multi-party systems is the
    freeze leader's. The caller compares the returned pid with the published one.
    """
    run = rep["dir"]
    main = os.path.join(run, "objects_host.jsonl")
    tables = [f for f in glob.glob(os.path.join(run, "objects_host*.jsonl"))
              if re.search(r"objects_host(_\d+|_party\d+)?\.jsonl$", f)]
    if rep.get("pid") and len(rep.get("traces") or {}) > 1:
        own = sorted(f for f in tables if _table_pid(f) == rep["pid"])
        if own:
            # objects_host.jsonl is a copy of one party's table on BumbleBee/BOLT/PUMA; prefer it
            # when it is the right party, so a single-table step keeps its existing path.
            return (main if main in own else own[0]), rep["pid"]
    if os.path.exists(main):
        return main, _table_pid(main)
    return None, None


def replicate_stats(reps):
    """(n, min, max, spread_pct) over replicate peaks; spread = (max-min)/median.

    Mixed-channel sets (a VmHWM trace and a polled-only trace) report count+extremes but NO spread:
    the difference is between channels, not runs (legacy SHAFT-GPU s0, spurious 39%). Same for n<2,
    where max==min would print 0.0% and read as "perfect" rather than "never tested".
    """
    if not reps:
        return 0, "", "", ""
    peaks = sorted(r["peak"] for r in reps)
    med = peaks[len(peaks) // 2]
    lo, hi = peaks[0], peaks[-1]
    if len(peaks) < 2:
        return len(peaks), lo, hi, ""
    if len({r["source"] for r in reps}) > 1:
        return len(peaks), lo, hi, ""
    return len(peaks), lo, hi, (round(100.0 * (hi - lo) / med, 3) if med else "")


def device_peak_from_marker(step_dir):
    """True when this step's device peak came from a `GPUPEAK|` marker, not the fallback.

    Separate from device_peak() so the caller uses it only to LABEL the number, never to branch
    measurement logic.
    """
    # sorted(): the per-party list is published in this order (per_party_reserved_kb), so an
    # unsorted glob makes a fresh clone's --check fail with no measurement changed (SHAFT-GPU-ViT).
    for f in sorted(glob.glob(os.path.join(step_dir, "logs/**/all_markers.txt"), recursive=True)):
        with open(f, errors="replace") as fh:
            for ln in fh:
                if ln.startswith("GPUPEAK|") and re.search(r"peak_reserved_kb=([0-9]+)", ln):
                    return True
    return False


def _run_key(step_dir, path):
    """The replicate a file under logs/ belongs to: the first path component below logs/."""
    rel = os.path.relpath(path, os.path.join(step_dir, "logs"))
    return rel.split(os.sep, 1)[0] if os.sep in rel else ""


def device_peak(step_dir):
    """Device peak: per run the maximum over the parties (both must fit), published as the MEDIAN
    over runs of that maximum, the same rule as the host peak (decision of 2026-09-29; until then
    the maximum over every run and party was published).

    Two sources with different units, so the second is a named fallback and the unit rides into
    evidence: GPUPEAK| peak_reserved_kb (RESERVED, torch / MOAI-GPU's snni_gpupeak), else cudarec's
    ALLOCATED bytes (SIGMA-GPU, which calls cudaMallocAsync with no pool). Empty column here would
    leave a two-pool system with no W and no derivable step.

    Returns (published_kb, every value in file order, per-run maxima sorted).
    """
    vals, byrun = [], {}
    for f in sorted(glob.glob(os.path.join(step_dir, "logs/**/all_markers.txt"), recursive=True)):
        with open(f, errors="replace") as fh:
            for ln in fh:
                if ln.startswith("GPUPEAK|"):
                    m = re.search(r"peak_reserved_kb=([0-9]+)", ln)
                    if m:
                        v = int(m.group(1))
                        vals.append(v)
                        k = _run_key(step_dir, f)
                        byrun[k] = max(byrun.get(k, 0), v)
    if not vals:
        for f in sorted(glob.glob(os.path.join(step_dir, "logs/**/objects_vram.jsonl"),
                               recursive=True)):
            try:
                with open(f, errors="replace") as fh:
                    for ln in fh:
                        ln = ln.strip()
                        if not ln:
                            continue
                        o = json.loads(ln)
                        kb = o.get("rss_kb")
                        if isinstance(kb, int) and kb > 0:
                            vals.append(kb)
                            k = _run_key(step_dir, f)
                            byrun[k] = max(byrun.get(k, 0), kb)
                        break
            except (ValueError, OSError):
                continue
    if not vals:
        return 0, vals, []
    runmax = sorted(byrun.values())
    return runmax[len(runmax) // 2], vals, runmax


def load_objects(step_dir, pool, run_dir=None, path=None):
    """Canonical decomposition for one pool: {label: bytes}, its total, and its provenance.

    Prefers the MEDIAN run's own table (else the step-level one), so peak and decomposition share
    one execution; a stale step-level table reads exactly like a current one. Provenance is carried
    because the unit can differ: pagemap = RESIDENT bytes, torch_state = ALLOCATED; never compared.
    `path` names one party's table explicitly (see host_table()).
    """
    if path is None and run_dir:
        cand = os.path.join(run_dir, f"objects_{pool}.jsonl")
        if os.path.exists(cand):
            path = cand
    if path is None:
        path = os.path.join(step_dir, f"objects_{pool}.jsonl")
    if not os.path.exists(path):
        return {}, 0, {}
    best, best_tot, best_rec = {}, -1, {}
    with open(path, errors="replace") as fh:
        for ln in fh:
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            objs = rec.get("objects", {})
            # Backstop only: a net-free site prints its signed counter unsigned (~2^64) and would
            # poison every share on the row. The real fix is at the producer (resolve_device.py
            # clamps net-free sites to zero); dropping here is the safe direction but signals a
            # producer bug, and is blunter since a key aggregates a site's rows (may hold real bytes too).
            _bogus = {k: v for k, v in objs.items() if isinstance(v, (int, float)) and v >= (1 << 63)}
            if _bogus:
                objs = {k: v for k, v in objs.items() if k not in _bogus}
                rec.setdefault("_dropped_sentinels", {}).update(
                    {k: "%#x" % int(v) for k, v in _bogus.items()})
            tot = sum(objs.values())
            if tot > best_tot:
                best, best_tot, best_rec = objs, tot, rec
    meta = {k: best_rec[k] for k in ("source", "sample_rss_kb", "snapshot_rss_kb", "sample_ts_ms",
                                     "drops", "_dropped_sentinels")
            if k in best_rec}
    # sample_rss_kb is the RSS the poller READ before it sent SIGSTOP. A peak that falls between that
    # read and the stop leaves a table taken lower than the header says (SIGMA-GPU s7: read at
    # 1,954,408 kB, frozen at 1,117,388 kB). poll_peak.sh records the frozen RSS as the third field of
    # peaks/SNAPSHOT_SPAN (ts trigger pre post pause); the alignment checks use the lower of the two.
    frozen = _frozen_rss(step_dir, run_dir, path, meta.get("sample_ts_ms")) if pool == "host" else None
    if frozen and meta.get("sample_rss_kb") and frozen < meta["sample_rss_kb"]:
        meta["polled_rss_kb"] = meta["sample_rss_kb"]
        meta["sample_rss_kb"] = frozen
    return best, max(best_tot, 0), meta


def _frozen_rss(step_dir, run_dir, table_path, ts):
    """Frozen RSS (kB) of the capture taken at `ts`, from any peaks/SNAPSHOT_SPAN of the run."""
    if not ts:
        return None
    roots = [d for d in (run_dir, os.path.dirname(table_path or "")) if d]
    if step_dir:
        roots += glob.glob(os.path.join(step_dir, "logs", "run*")) + [os.path.join(step_dir, "logs")]
    for root in roots:
        for span in glob.glob(os.path.join(root, "**", "SNAPSHOT_SPAN"), recursive=True):
            for ln in open(span, errors="replace"):
                p = ln.split()
                if len(p) >= 3 and p[0] == str(ts) and p[2].isdigit():
                    return int(p[2])
    return None


def frozen_ms(step_dir, run_dir=None):
    """Milliseconds the process was held stopped (SIGSTOP) for its captures.

    Freeze makes objects/mappings/libraries describe one instant; its cost is wall time, known
    exactly, subtracted from wall before runtime_delta_pct so a step is not typed free/paid partly
    by measurement pauses. Max over the run's traces = the freeze leader's (followers carry no total).
    """
    base = run_dir or os.path.join(step_dir, "logs")
    best = 0
    for f in glob.glob(os.path.join(base, "**", "vmrss.log"), recursive=True):
        with open(f, errors="replace") as fh:
            for ln in fh:
                if ln.startswith("#"):
                    m = re.search(r"frozen_ms=([0-9]+)", ln)
                    if m:
                        best = max(best, int(m.group(1)))
    # A census watcher attached from outside the poller (ARION a3_gogc25_direct, srun --overlap,
    # arion/census_watch_lowmin.sh) stops the process the same way but writes its pauses to its own
    # log, so they are not in the poller's frozen_ms and are added here.
    for f in glob.glob(os.path.join(base, "**", "census_watch.log"), recursive=True):
        with open(f, errors="replace") as fh:
            best += sum(int(m) for m in re.findall(r"pause_ms=([0-9]+)", fh.read()))
    return best


def wall_spread_pct(step_dir):
    """(n, spread_pct) over the replicates' wall times, or (0, "") for fewer than two.

    The runtime axis's own measured noise floor: the replicates' wall spread IS the free/paid
    threshold, measured on the same step and machine. Taken over CORRECTED walls (matching the
    delta), or the classification biases toward paid (SHARK baseline: raw 2.73% vs corrected 3.97%).
    """
    walls = corrected_walls(step_dir)
    if len(walls) < 2:
        return len(walls), ""
    med = statistics.median(walls)
    return len(walls), (round(100.0 * (max(walls) - min(walls)) / med, 3) if med else "")


def program_wall_seconds(run_dir):
    """The program's own wall clock, for a combined-channel ARION run, else None.

    In the combined arrangement the batch script parsed the Go heap dump after the program ended and
    counted that post-processing into wall_seconds.txt until the script was fixed on 2026-09-28 (it
    now stops the clock at the program's end). The parse took ~950 s on a0 and ~1,230 s on a1 and a
    few seconds elsewhere, so the raw file would type steps by the size of a dump. ARION prints its own
    total ("Wall-Clock Total: 10h15m25.69s") in stdout.log; that line is the same measure on every
    run, before and after the fix (a4_threads16: 36,926 s against 36,928 s in wall_seconds.txt).
    """
    meta = os.path.join(run_dir, "RUN_META")
    out = os.path.join(run_dir, "stdout.log")
    try:
        if "combined_channel=1" not in open(meta, errors="replace").read():
            return None
        m = re.search(r"Wall-Clock Total:\s*(?:(\d+)h)?(?:(\d+)m)?([0-9.]+)s",
                      open(out, errors="replace").read())
    except OSError:
        return None
    if not m:
        return None
    h, mi, sec = int(m.group(1) or 0), int(m.group(2) or 0), float(m.group(3))
    return int(round(h * 3600 + mi * 60 + sec))


def wall_seconds(step_dir, run_dir=None):
    """This step's wall clock, from ONE named run.

    run_dir is required whenever the value is paired with another per-run quantity: globbing returns
    an arbitrary run and would subtract one run's freeze from another's wall (the 2026-07-30 defect,
    15 of 22 steps), an error the size of the 1-16% wall spread it feeds.
    """
    if run_dir:
        pw = program_wall_seconds(run_dir)
        if pw is not None:
            return pw
        f = os.path.join(run_dir, "wall_seconds.txt")
        try:
            return int(open(f).read().strip())
        except (OSError, ValueError):
            return None
    f = os.path.join(step_dir, "logs", "wall_seconds.txt")
    if os.path.exists(f):
        try:
            return int(open(f).read().strip())
        except ValueError:
            pass
    for f in sorted(glob.glob(os.path.join(step_dir, "logs/**/wall_seconds.txt"), recursive=True)):
        try:
            return int(open(f).read().strip())
        except ValueError:
            pass
    return None


def corrected_walls(step_dir):
    """Each replicate's program time = its own wall minus its OWN freeze total (seconds).

    Paired by run (a freeze belongs to its run) and kept as all replicates, because cost_type judges
    runtime_delta_pct against the all-run spread; a single-run delta compares two kinds of number
    (SHARK s1: single-run -8.3% vs median-of-three -1.4%). The freeze correction scales with the peak.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(step_dir, "logs", "run*"))):
        if not os.path.isdir(d):
            continue
        w = wall_seconds(step_dir, d)
        if w is None:
            continue
        out.append(w - frozen_ms(step_dir, d) / 1000.0)
    if not out:
        # A system that declared `replicates: 1`, or a step harvested before the run layout.
        w = wall_seconds(step_dir)
        if w is not None:
            out.append(w - frozen_ms(step_dir) / 1000.0)
    return out


def levers(step_dir):
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return ""
    return " ".join(l.strip() for l in open(f) if l.strip() and not l.startswith("#"))


def impl_fingerprint(step_dir):
    """Artifact fingerprint that lets a reviewer confirm only the lever changed between two steps.

    Three per-system lever mechanisms, uniform record: env-gated (constant fingerprint, SHAFT-GPU),
    rebuilt (one src_diff per step, LLAMA dealer), image-per-step (one digest, SHARK/PUMA/BOLT/...).
    """
    # Meta is per run (logs/run<N>/RUN_META); reading only logs/RUN_META left this empty for every
    # replicated system, silently disabling the artifact-differs-from-parent check (found 2026-07-30).
    meta = os.path.join(step_dir, "logs", "RUN_META")
    if not os.path.exists(meta):
        cands = sorted(glob.glob(os.path.join(step_dir, "logs", "**", "RUN_META"), recursive=True))
        meta = cands[0] if cands else meta
    if not os.path.exists(meta):
        return ""
    txt = open(meta, errors="replace").read()
    m = re.search(r"src_diff_md5=(\S+)", txt)
    if m:
        return "src_diff_md5:" + m.group(1)
    m = re.search(r"^(?:SIF|image)=(\S+)", txt, re.M)
    if m:
        return "image:" + os.path.basename(m.group(1))
    return ""


def within_tolerance(xs, ys, tol_rel, tol_abs):
    """Are two numeric observables equal at a declared tolerance? -> (ok, worst, why)

    One implementation for both T2 comparisons (replicates vs each other, step vs baseline).
    Tolerance is exactly one of gate_tolerance (relative) or gate_tolerance_abs (absolute) from
    MANIFEST; an undeclared tolerance is refused, not defaulted.
    """
    if len(xs) != len(ys):
        return False, None, f"{len(xs)} values against {len(ys)}"
    if (tol_rel is None) == (tol_abs is None):
        return False, None, ("declare exactly one of gate_tolerance (relative) or "
                             "gate_tolerance_abs (absolute); neither or both is not a tolerance")
    worst = 0.0
    for x, y in zip(xs, ys):
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False, None, "the observable contains values that are not numbers"
        d = abs(fx - fy) if tol_abs is not None else abs(fx - fy) / max(abs(fy), 1e-12)
        worst = max(worst, d)
    tol = tol_abs if tol_abs is not None else tol_rel
    unit = "absolute" if tol_abs is not None else "relative"
    return worst <= tol, worst, f"worst {unit} difference {worst:.3e} against tolerance {tol:.3e}"


def read_observable(d):
    """One replicate's gate observable, in FILE ORDER, or None if it has none."""
    for fname in ("gate.txt", "keys.md5"):
        f = os.path.join(d, fname)
        if os.path.exists(f):
            toks = [l.split()[0] for l in open(f, errors="replace") if l.split()]
            if toks:
                return toks
            break
    return None


def gate_degenerate(step_dir):
    """(n, value) if this step's observable is one value repeated, else None.

    An observable that exists is not one that discriminates: 98,304 exact zeros reproduce perfectly
    and cannot fail (SHARK's first gate, MOAI-GPU's values -> T1s, LLAMA online's baseline). Narrow:
    only a long-enough numeric observable, so a hash list or structural trace is not flagged.
    """
    for f in sorted(glob.glob(os.path.join(step_dir, "logs/**/gate.txt"), recursive=True)):
        # The header is not the observable: GATE|logits|n=98304|... carries its length before the
        # values, so counting it makes an all-zeros file look like two values. Take the last field.
        vals = []
        for line in open(f, errors="replace"):
            vals += re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line.rsplit("|", 1)[-1])
        if len(vals) < 32:
            continue
        if len(set(vals)) == 1:
            return len(vals), vals[0]
        return None
    return None


def correctness_gate(step_dir, tier=None, tol_rel=None, tol_abs=None, med_dir=None):
    """The system's own output-equality gate for this step, compared against the BASELINE (not an
    external reference: that tests the system, this tests whether the step changed anything).

    Observable is hashed into logs/gate.txt (dealers: keys.md5). Tiers (MANIFEST gate_tier):
    T1 byte-identical, T1s structural (only after a discrimination test, row states what it misses),
    T2 numeric within a declared tolerance, T3 label-only, T0 none (declared). A gate-failing step
    changed the computation and is not a waterfall step.
    """
    # Replicates are also the determinism check: disagreeing runs => NONDETERMINISTIC, build fails
    # (SHAFT-GPU seeded from os.urandom, 5 runs spanned 3.8x). At T2 "disagree" means beyond the
    # declared tolerance, not not-byte-identical; the tier is consulted here, before any baseline
    # comparison, so a protocol-approximate system reproduces itself within its declared tolerance.
    rep_dirs = sorted(glob.glob(os.path.join(step_dir, "logs", "run*")))
    if tier == "T2":
        obs = [(d, read_observable(d)) for d in rep_dirs]
        obs = [(d, o) for d, o in obs if o]
        if obs:
            # Every pair, not each against the first: agreement is a property of the set.
            for i in range(len(obs)):
                for j in range(i + 1, len(obs)):
                    ok, _, _ = within_tolerance(obs[i][1], obs[j][1], tol_rel, tol_abs)
                    if not ok:
                        return "NONDETERMINISTIC"
            # Publish the median replicate's values, matching the run whose peak/table are published.
            chosen = dict(obs).get(med_dir) or obs[0][1]
            return " ".join(chosen)
    else:
        vals = []
        for d in rep_dirs:
            toks = read_observable(d)
            if toks:
                vals.append(" ".join(sorted(toks)))
        if vals:
            return vals[0] if len(set(vals)) == 1 else "NONDETERMINISTIC"

    # n=1 step. A T2 observable stays in file order (sorting pairs 9.9e-01 with 1.0e+00).
    toks = read_observable(os.path.join(step_dir, "logs"))
    if toks:
        return " ".join(toks if tier == "T2" else sorted(toks))
    return ""


def group_sites(objs):
    """Sum allocation sites of the same function into one object; file:/anon: pass through untouched.

    A symboliser splits one object across several file:line rows and can rank it below something
    half its size. Grouped by FUNCTION (the unit that allocates/frees one thing), not file. Measured:
    LLAMA d4 SlothGelu was 5x34.6 per-site but 173.2 grouped and won, and the step measured -169136 kB.
    """
    out = {}
    for k, v in objs.items():
        m = re.match(r"heap:[^:]+:\d+(?: \(discriminator \d+\))?:(.+)", k)
        key = f"heap:{m.group(1)}" if m else k
        out[key] = out.get(key, 0) + v
    return out


def gate_equal(a, b, tier, tol, tol_abs=None):
    """Does this step's gate observable match the baseline's at the declared tier?

    T1/T3 are exact; T2 allows a MANIFEST-declared tolerance (relative or absolute, exactly one) and
    reports by how much, since a bare FAIL cannot tell a reordering from a broken protocol. Undeclared
    tolerance is treated as exact. Uses within_tolerance (same fn as the replicate check).
    """
    if a == b:
        return True, ""
    if tier != "T2":
        return False, ""
    ok, _, why = within_tolerance(a.split(), b.split(), tol, tol_abs)
    return ok, why


def declared_object(step_dir):
    """The object this step attacks, declared in levers.env as `# object_attacked: <site substring>`.

    Machine-checks the object half of "largest object with a free fix" (the pool half was already
    checked). Declared, not derived: a measured drop cannot say which object a step aimed at.
    """
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return ""
    m = re.search(r"object_attacked\s*[:=]\s*(.+)", open(f, errors="replace").read())
    return m.group(1).strip() if m else ""


def declared_runtime_knob(step_dir):
    """Whether this step declares its mechanism is a runtime memory knob (`# runtime_knob: yes`).

    Opt-in exception admitting a retention aggregate (go:/sealpool:) as a target when the step's
    lever IS the declared parameter that sets its size (GOGC). Declared, not inferred: from the table
    alone a knob step looks identical to one that changes the allocator while claiming an object.
    """
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return False
    return bool(re.search(r"^#\s*runtime_knob\s*[:=]\s*yes\s*$",
                          open(f, errors="replace").read(), re.M | re.I))


def declared_transcript_artefact(step_dir):
    """Declared exception to phase B's "same bytes", as `# transcript_artefact: <bytes> <file>`.
    Returns (bytes, path) or (None, None).

    Phase B needs identical transcript bytes (it retracted BumbleBee x7: -28.9% with +41% bytes).
    The exception fits SHAFT-GPU s5_per_layer, which REMOVES a 524,288-byte ONNX-export artefact that
    predates the line (it also separates the SHAFT-CPU/GPU baselines with no lever). Conditions (none
    a threshold): declared in advance with the exact count; count equals the measured difference; the
    difference is a REDUCTION (adding traffic is never admitted); the evidence file exists in-step.
    """
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return None, None
    m = re.search(r"^#\s*transcript_artefact\s*[:=]\s*(\d+)\s+(\S+)\s*$",
                  open(f, errors="replace").read(), re.M)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def declared_prerequisite(step_dir):
    """Declared exception to rule 6's "free before paid", two levers.env lines (`# requires:` and
    `# refuted_as: <dir> <peak_before> <peak_after>`). Returns (requires, refuted_dir, before, after).

    Cost class is a property of (lever, tree), not the lever: MOAI-GPU measured the same edit at +2.95%
    on s3 and -12.52% on s9, so a free fix can legitimately follow the paid step that first exposed it.
    Conditions (none a threshold): names an earlier PAID step it requires; names a same-lever off-path
    run with its two peaks that did NOT reduce (peak_after>=peak_before); that dir exists with RESULT.md.
    Cannot check that the prerequisite is the EARLIEST one (argued from the marker series in the step's text).
    """
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return None, None, None, None
    txt = open(f, errors="replace").read()
    mreq = re.search(r"^#\s*requires\s*[:=]\s*(\S+)\s*$", txt, re.M)
    mref = re.search(r"^#\s*refuted_as\s*[:=]\s*(\S+)\s+(\d+)\s+(\d+)\s*$", txt, re.M)
    if not mreq or not mref:
        return None, None, None, None
    return mreq.group(1), mref.group(1), int(mref.group(2)), int(mref.group(3))


def declared_skips(step_dir):
    """Larger objects deliberately passed over, declared as `# object_skipped: <name>`.

    The rule is "largest object WITH A FREE FIX" (free-fix-ness is read from source, not the table),
    so a smaller target is legitimate but must be declared and dispositioned in off-path.md; a silent
    small attack still fails. SHARK's largest object is 94.8% of peak with only a paid fix (+26.9%).
    """
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return []
    txt = open(f, errors="replace").read()
    return [m.strip() for m in re.findall(r"object_skipped\s*[:=]\s*(.+)", txt)]


def declared_enabler(step_dir):
    """The paid ENABLER a free-after-paid step names as `# not_selectable_until: <step_id>`.

    Second admission to rule 6, distinct from `# refuted_as:`: a free fix whose OBJECT was invisible
    in the table until an earlier paid step exposed it (MOAI-CPU's key bank is dead at the stock-pool
    peak and only becomes a live, nameable object once the threshold pool is in place). The object
    check below proves it: the attacked object is absent from the enabler's predecessor's table and
    present in the enabler's.
    """
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return None
    m = re.search(r"^#\s*not_selectable_until\s*[:=]\s*(\S+)\s*$",
                  open(f, errors="replace").read(), re.M)
    return m.group(1) if m else None


def identifies(att, table):
    """The one object in `table` whose full key contains `att`, or None if zero or several do.

    A declaration names an object by a substring that must pick exactly one row, which is what makes
    the name discriminating. It handles the recording-pool tables where the program-frame half is a
    generic wrapper (`all_layer_test()` wraps the whole run) so the ALLOCATOR half is what
    discriminates, e.g. `create_galois_keys`; there `caller_half` is non-unique and this is not.
    """
    if not att or not table:
        return None
    hits = [k for k in table if att in k]
    return hits[0] if len(hits) == 1 else None


# Retention aggregates, by prefix (admitted only via the runtime_knob branch below). "sealpool:"
# not "seal:", which is a prefix of the seal:: namespace and would refuse a good object declaration.
RETENTION_PREFIXES = ("go:", "sealpool:")


def caller_half(key):
    """The discriminating (caller) half of an object key, which a declaration must name.

    Keys of the form `caller <- allocating function` return the LEFT side, because the right side is
    the allocator the pair exists to get past and matching it is vacuous (a pool-only declaration
    matches every SEAL row at depth 5). A key with no pair is returned unchanged.
    """
    return key.split(" <- ", 1)[0] if " <- " in key else key


def read_transcript(step_dir, run_dir=None):
    """(bytes, rounds) exchanged by party 0, from `TRANSCRIPT|party=0|bytes=<n>|rounds=<n>`.

    Read from the median run (one execution). Added because CrypTen's own "comm byte: 10.46 GB" is
    5 MB resolution and cannot support phase B's "same bytes" claim.
    """
    dirs = [run_dir] if run_dir else []
    dirs.append(os.path.join(step_dir, "logs"))
    for d in dirs:
        if not d:
            continue
        for f in sorted(glob.glob(os.path.join(d, "**", "*.log"), recursive=True)) + \
                 sorted(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)):
            try:
                txt = open(f, errors="replace").read()
            except OSError:
                continue
            m = re.search(r"TRANSCRIPT\|party=0\|bytes=(\d+)\|rounds=(\d+)", txt)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None, None


def declared_evidence(step_dir):
    """The evidence class this step claims (`exact` or `equivalent`), from levers.env.

    Declared, not derived: a moved output could be a repartition or a broken protocol. Phase B is
    granted only to a step that asked for it in advance and satisfied both conditions.
    """
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return ""
    m = re.search(r"evidence\s*[:=]\s*(exact|equivalent)", open(f, errors="replace").read())
    return m.group(1) if m else ""


def declared_plaintext_test(step_dir):
    """The plaintext equivalence test cited as `# plaintext_equivalence: <path>` (resolved in-step).

    Phase B's second condition: the transcript shows same bytes, this shows same function. Neither
    alone suffices (same traffic + different arithmetic, or the reverse, which x7 was).
    """
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return "", False
    m = re.search(r"plaintext_equivalence\s*[:=]\s*(\S+)", open(f, errors="replace").read())
    if not m:
        return "", False
    rel = m.group(1)
    return rel, os.path.exists(os.path.join(step_dir, rel))


def declared_pool_skips(step_dir):
    """Pools deliberately passed over, declared as `# pool_skipped: host`.

    The pool choice is a reporting convention, not forced; skipping the larger-W pool is allowed but
    must be declared and dispositioned in off-path.md (undeclared still fails). Needed at SHAFT-GPU s1
    (W_host 10.78 vs W_vram 10.01) where the host side had nothing left to attack (object_vs_unnamed < 1),
    so forcing it would have required an inert step.
    """
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return []
    txt = open(f, errors="replace").read()
    return [m.strip().lower() for m in re.findall(r"pool_skipped\s*[:=]\s*(vram|host)", txt)]


def declared_cost_class(step_dir):
    """The cost class this step claims, declared as `# cost_class: free`.

    Only consulted when the measurement cannot decide (replicates:1, so no wall spread); otherwise the
    measured classification wins and this is ignored. Exists so a +28.7% step is not published with an
    empty cost_type (an unstated claim that rule 6's free-before-paid check would silently skip); the
    row records that the class was declared rather than measured.
    """
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return ""
    txt = open(f, errors="replace").read()
    m = re.findall(r"cost_class\s*[:=]\s*(free|paid)", txt)
    return m[0].lower() if m else ""


def declared_phase_effect(step_dir):
    """The phase marker a plateau step claims to lower, declared as `# phase_effect: <marker>`.

    On a system whose peak is a plateau of phases within the spread (MOAI-CPU after the phase split:
    four bootstrap phases within 0.7 GiB), a step that removes the largest object with a free fix
    lowers ITS phase while the global peak moves to the next one. The declared marker names the
    phase; it must be a marker at which the run's high-water so far IS that phase's maximum (the
    first phase that sets a new high, e.g. after_attention), because markers.txt carries the
    cumulative high-water, not an in-phase maximum.
    """
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return ""
    m = re.search(r"^#\s*phase_effect\s*[:=]\s*(\S+)\s*$", open(f, errors="replace").read(), re.M)
    return m.group(1) if m else ""


def marker_hwm_kb(step_dir, marker):
    """The high-water (hwm_kb) recorded at an occurrence of `MEM|<marker>|...` in the median run's
    markers.txt, or None. `<marker>` alone means the FIRST occurrence; `<marker>@<n>` the n-th
    (1-based), for a phase that repeats per layer and whose lever acts in a later layer only
    (MOAI-CPU m12: the attention keys are released in the last layer, so the effect is in the
    second occurrence of the bootstrap markers)."""
    reps = sorted(glob.glob(os.path.join(step_dir, "logs", "run*")))
    if not reps:
        return None
    f = os.path.join(reps[len(reps) // 2], "markers.txt")
    if not os.path.exists(f):
        return None
    want = 1
    if "@" in marker:
        marker, n = marker.rsplit("@", 1)
        want = int(n) if n.isdigit() and int(n) >= 1 else 1
    seen = 0
    for ln in open(f, errors="replace"):
        if ln.startswith(f"MEM|{marker}|"):
            seen += 1
            if seen == want:
                m = re.search(r"hwm_kb=(\d+)", ln)
                return int(m.group(1)) if m else None
    return None


def declared_pool(step_dir):
    """Which pool this step was designed to attack, declared as `# pool_attacked: vram`.

    Declared, not derived: a measured drop cannot tell the attacked pool from one that fell as a side
    effect (discarded SHAFT-GPU s2 was a VRAM lever whose host pool also fell).
    """
    f = os.path.join(step_dir, "levers.env")
    if not os.path.exists(f):
        return ""
    m = re.search(r"pool_attacked\s*[:=]\s*(vram|host)", open(f, errors="replace").read())
    return m.group(1) if m else ""


def _step_order(path):
    """Sort key for step directories: numeric on the leading index, then the rest as text.

    Plain sorted() puts s11_ between s0_ and s2_, so pool_required/object_required (the PREVIOUS
    step's) compare against the wrong predecessor. Found on MOAI-GPU s11 (first two-digit step).
    """
    b = os.path.basename(path)
    m = re.match(r"([a-z]+)(\d+)(.*)", b)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (b, -1, "")


def collect(sysdir):
    targets, overheads = read_manifest(sysdir)
    steps = sorted((d for d in glob.glob(os.path.join(sysdir, "steps", "*")) if os.path.isdir(d)),
                   key=_step_order)
    rows, per_step = [], {}
    prev_v = prev_h = prev_w = None
    prev_required = None
    prev_object = ""
    prev_sd = None
    # PHASE ORDERING (README.md, "Two phases of evidence"). Exact evidence is exhausted
    # before equivalence evidence is admitted, for the same reason free is exhausted before
    # paid: once the weaker claim is allowed, everything after it is easier to justify.
    prev_comm_bytes = prev_comm_rounds = None
    seen_equivalent = None
    # The previous step's top object in EACH pool (not only the one it attacked): the requirement is
    # the largest attackable object at the previous peak in the pool THIS step attacks, so when the
    # pool flips (SHAFT-GPU s1->s2) the successor is judged against the table it was chosen from.
    prev_top = {}
    prev_grouped = {}
    base_gate = None
    # Steps failed because the peak did not move, kept apart from steps failed for attacking the
    # wrong pool. Both set conformant="no", and reporting them under one heading told a reader to
    # go and look at the pool choice of a step whose pool choice was correct. README.md:
    # "a check that reports the WRONG REASON is worse than one that fails loudly".
    inert_ids = set()

    for sd in steps:
        sid = os.path.basename(sd)
        r = {c: "" for c in SCHEMA}
        r["step_id"] = sid
        r["lever"] = levers(sd)
        r["impl_fingerprint"] = impl_fingerprint(sd)

        vram, per_party, vram_runs = device_peak(sd)
        host, host_polled, host_src = poll_peak(sd)
        reps = host_replicates(sd)
        # The MEDIAN replicate is the run whose peak is published, so anything else describing
        # "this step's run" -- its freeze total, its object table -- has to come from that same run
        # or it describes a different execution. Same selection as poll_peak().
        med_rep = reps[len(reps) // 2] if reps else None
        med_dir = med_rep["dir"] if med_rep else None
        n_runs, rep_min, rep_max, rep_spread = replicate_stats(reps)
        r["n_runs"] = n_runs
        r["peak_host_min"] = rep_min
        r["peak_host_max"] = rep_max
        r["spread_host_pct"] = rep_spread
        # The other parties' peaks of the SAME median run, largest first entry only, beside the
        # published maximum. Raw values (no instrument subtraction); at the 3 MB scale of the
        # recorder table that difference is far below what this column exists to show.
        if med_rep and med_rep.get("other_peaks"):
            r["peak_host_other"] = max(med_rep["other_peaks"])
            r["peak_party"] = med_rep["party"]
            if not med_rep.get("leader_known", True):
                note = ("freeze leader not identifiable from this run's traces, so the published "
                        "party (the heavier) cannot be named as leader or non-leader")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
        instr = instrument_kb(sd)
        if instr and host:
            host -= instr
            host_polled = max(0, host_polled - instr)
        r["peak_vram"] = vram or ""
        r["peak_host"] = host or ""
        if per_party:
            # The label is the UNIT, not decoration. `reserved` is a pooling allocator's high-water;
            # `allocated` is what a program calling the CUDA allocator directly ever held at once.
            # A reader comparing a VRAM peak across systems has to know which of the two it is, and
            # the only place that survives into the published row is here.
            unit = "reserved" if device_peak_from_marker(sd) else "allocated"
            r["evidence"] = f"per_party_{unit}_kb=" + "/".join(str(v) for v in per_party)
            if unit == "allocated":
                r["evidence"] += ("; device peak is ALLOCATED bytes from the cudarec table rather than a "
                                  "pooling allocator's reserved high-water: this program calls the "
                                  "CUDA allocator directly and no pool holds anything back, so "
                                  "allocated is the only device quantity it has. Not comparable "
                                  "with a reserved figure from another system")
        if instr:
            note = (f"instrument subtracted: {instr} kB of recorder table removed from the host "
                    f"peak, so this single run reports what the program alone held")
            r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
        # A poller that fell short of the kernel's high-water mark did not see the peak instant,
        # so everything sampled AT the peak describes a lower moment. Say so, with the size of it.
        if host_src == "VmHWM" and host and host_polled and host > host_polled:
            miss = 100.0 * (host - host_polled) / host
            if miss >= 1.0:
                note = (f"host peak {host} kB is the kernel's VmHWM; the 100 ms poller only "
                        f"observed {host_polled} kB ({miss:.1f}% low), so this peak is a spike "
                        f"and the composition snapshot is not taken at it")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

        # Replicate count is declared and checked, not inferred from runs on disk. Three is the
        # floor; a system that cannot afford it (MOAI-CPU ~144 h/run) declares a lower n in MANIFEST.
        want = targets.get("replicates")
        want = int(want) if want not in (None, "") else 3
        if n_runs and n_runs < want:
            note = (f"{n_runs} run(s) on disk but the manifest calls for {want}: the spread "
                    f"is not established for this step")
            r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
        if n_runs > 1 and rep_spread == "":
            note = (f"{n_runs} runs found but measured by different channels (VmHWM vs polled), "
                    f"so their span {rep_min}-{rep_max} kB is not a run-to-run spread")
            r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

        objs, objs_meta = {}, {}
        for pool in ("vram", "host"):
            objs[pool], _, objs_meta[pool] = load_objects(sd, pool, med_dir)
        # The host decomposition belongs to one party. It is the published (heavier) party's own
        # table from the published run where that run carries one. Where it does not (only one
        # party's table was derived), the row keeps the decomposition it had under the freeze-leader
        # convention: the table of the run that is the median by the leader's peak. The ordering
        # decisions of the line were made on those tables, and a different run's table can rank
        # other objects first (SHAFT-CPU s4 run1: the leader's table sits 7.9% below the heavier
        # party's peak, in another phase). The row declares the mismatch, and every share of the
        # peak computed from the table (unattributed_host, named_frac_host, the alignment checks,
        # peak-objects.md) is taken against the decomposed party's own peak in that run, the instant
        # the table describes. W, the deltas and the spread stay on the published maximum.
        host_dec, dec_run = host, ""
        if med_rep and len(med_rep.get("traces") or {}) > 1:
            tpath, tpid = host_table(med_rep)
            dec_rep = med_rep
            # The heavier party's own table stands only if it describes that party's peak: the
            # same 10% limit that refuses a MISALIGNED decomposition (PROCEDURE, "Measurement"),
            # read from the table's own instant (the resident sample, else its smaps snapshot, which
            # is what the Python-census tables carry). A non-leader is sampled without the freeze,
            # and SHAFT-CPU s5 run3's non-leader table sits 10.8% below its peak.
            if tpath and tpid == med_rep["pid"]:
                gap = _table_gap_pct(tpath, med_rep["traces"].get(tpid))
                if gap is not None and gap >= MISALIGN_FAIL_PCT:
                    note = (f"the {med_rep['party']}'s own object table was sampled {gap:.1f}% below "
                            f"its peak, beyond the {MISALIGN_FAIL_PCT:.0f}% limit for a decomposition "
                            f"of that peak, so it is not used")
                    r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
                    tpid = None
                    # Another replicate whose published party's own table IS aligned describes the
                    # same step at its peak (a short spike can fall before the freeze in one run and
                    # be caught in another). The nearest such replicate by peak is used, and said so.
                    _alts = []
                    for _rep in reps:
                        if _rep is med_rep or len(_rep.get("traces") or {}) < 2:
                            continue
                        _ap, _apid = host_table(_rep)
                        if _ap and _apid == _rep["pid"]:
                            _ag = _table_gap_pct(_ap, _rep["traces"].get(_apid))
                            if _ag is not None and _ag < MISALIGN_FAIL_PCT:
                                _alts.append((abs(_rep["peak"] - med_rep["peak"]), _rep, _ap, _apid, _ag))
                    if _alts:
                        _alts.sort(key=lambda x: x[0])
                        _d, alt_rep, tpath, tpid, _ag = _alts[0]
                        dec_rep = alt_rep
                        note = (f"the decomposition is the {alt_rep['party']}'s own table from "
                                f"{os.path.basename(alt_rep['dir'])}, whose frozen capture lies "
                                f"{_ag:.1f}% below that party's own peak in that run")
                        r["evidence"] = f"{r['evidence']}; {note}"
            if dec_rep is med_rep and tpid != med_rep["pid"]:
                by_leader = sorted(reps, key=lambda x: x["traces"].get(x["leader_pid"], x["peak"]))
                dec_rep = by_leader[len(by_leader) // 2]
                tpath = os.path.join(dec_rep["dir"], "objects_host.jsonl")
                tpid = _table_pid(tpath) if os.path.exists(tpath) else None
            if tpath and os.path.exists(tpath):
                objs["host"], _, objs_meta["host"] = load_objects(sd, "host", dec_rep["dir"], tpath)
            if dec_rep is med_rep and tpid == med_rep["pid"]:
                r["decomposed_party"] = med_rep["party"]
                cap = _maxparty_capture(med_rep["dir"], tpid)
                if cap is not None:
                    # freeze_mode=maxparty: at each new high of the maximum over the parties the
                    # party holding it is captured with every party stopped, so the published
                    # party's table is atomic whether or not it is the leader, and its last capture
                    # is at its peak. CAPTURE names the party that held the maximum; a mismatch
                    # would mean the published party's peak was a spike the poller did not see,
                    # whose instant the 10% check above has already judged.
                    note = ("host decomposition is the published party's own table from the frozen "
                            "max-over-parties capture (freeze_mode=maxparty)")
                    if cap.get("max_pid") and cap["max_pid"] != tpid:
                        note += (f"; its last capture was taken while the other party held the "
                                 f"maximum ({cap.get('max_rss_kb')} kB)")
                    r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
                elif med_rep["party"] == "non-leader":
                    note = ("host decomposition is the non-leader's own table, sampled at its own "
                            "peak by its poller without the freeze (only the leader's captures are "
                            "atomic)")
                    r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            elif tpid in dec_rep["traces"]:
                r["decomposed_party"] = ("leader" if tpid == dec_rep.get("leader_pid")
                                         else "non-leader")
                raw = dec_rep["traces"][tpid]
                dec_run = os.path.basename(dec_rep["dir"])
                host_dec = raw - instr if instr else raw
                where = ("this run" if dec_rep is med_rep else
                         f"{os.path.basename(dec_rep['dir'])} "
                         f"(the published peak is {os.path.basename(med_dir)}'s)")
                note = (f"PARTY MISMATCH: the published host peak {host} kB is the "
                        f"{med_rep['party']}'s (maximum over the parties), which has no usable object "
                        f"table of its own; the decomposition is the {r['decomposed_party']}'s "
                        f"table from {where}, at that party's own peak of {host_dec} kB. "
                        f"unattributed_host and named_frac_host describe that table against that "
                        f"peak")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

        for pool, val in (("vram", vram), ("host", host)):
            tgt = targets.get(pool)
            ovh = overheads.get(pool)
            if ovh is None:
                ovh = measured_overhead(objs[pool])
            r[f"overhead_{pool}"] = ovh
            # A mapped file that is not loaded code is the program's own data: it stays in the peak
            # and is reported (a 433 MB model blob was once subtracted as library text on 5 rows).
            _data_maps = sorted(((v, k) for k, v in objs[pool].items()
                                 if k.startswith("file:") and not is_loaded_code(k)),
                                reverse=True)
            _data_kb = sum(v for v, _ in _data_maps) // 1024
            if _data_kb and val and _data_kb * 20 > val:
                note = (f"{_data_kb} kB of the {pool} peak is a file the PROGRAM mapped rather than loaded "
                        f"code, largest '{_data_maps[0][1][5:60]}' at "
                        f"{_data_maps[0][0] // 1048576} MB: counted in the peak and NOT deducted as "
                        f"overhead")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            if tgt and val:
                r[f"w_{pool}"] = round(max(0, val - ovh) / tgt, 2)

        if prev_v is not None and vram:
            r["delta_vram"] = vram - prev_v
        if prev_h is not None and host:
            r["delta_host"] = host - prev_h

        # Lowering one pool while raising the other is a PLACEMENT (cross_pool=yes), a fact from two
        # opposite-signed deltas. But the rise must exceed the rising pool's own spread, or the sign
        # is undetermined: SHAFT-GPU s1's host delta came out -369,820 kB and +119,484 kB on two
        # replicate sets. A rise inside the spread is reported in evidence, not the column.
        dv, dh = r["delta_vram"], r["delta_host"]
        if isinstance(dv, int) and isinstance(dh, int) and dv * dh < 0:
            rise, rise_peak, rise_pool = ((dh, host, "host") if dh > 0 else (dv, vram, "vram"))
            sp = rep_spread if rise_pool == "host" and isinstance(rep_spread, (int, float)) else 0.0
            rise_pct = (100.0 * rise / rise_peak) if rise_peak else 0.0
            if sp and rise_pct < sp:
                r["cross_pool"] = "no"
                note = (f"{rise_pool} rose {rise} kB, {rise_pct:.2f}% of its peak, inside this "
                        f"step's own {sp}% spread over {n_runs} runs: the other pool is not shown "
                        f"to have risen, so this is not a placement")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            else:
                r["cross_pool"] = "yes"
        elif isinstance(dv, int) or isinstance(dh, int):
            r["cross_pool"] = "no"

        # Inert-step guard, judged on the pool the step declared (pool_attacked), against that pool's
        # own measured spread. A |move| below the spread, in either sign, means the peak did not move
        # and is not a step. Device spread is from the replicates' own reserved high-water.
        pool = (declared_pool(sd) or "host").strip().lower()
        if pool == "vram" and isinstance(dv, int) and vram:
            d_used, peak_used, pool_word = dv, vram, "vram"
            # Over the replicates' per-run maxima, like the host spread: (max - min) / median.
            # Zero spread counts only with >=2 replicates (identical high-water = real zero). With
            # one replicate spread is left empty, not an invented 0.0, so a rise routes to the
            # one-replicate block below (MOAI-GPU and SIGMA-GPU run n=1 on this pool).
            vv = vram_runs
            spread_used = (round(100.0 * (vv[-1] - vv[0]) / vv[len(vv) // 2], 3)
                           if len(vv) > 1 and vv[len(vv) // 2] else "")
        else:
            d_used, peak_used, pool_word, spread_used = dh, host, "host", rep_spread
            if spread_used in ("", None) and targets.get("spread_host_pct_declared") is not None:
                spread_used = targets["spread_host_pct_declared"]
                note = (f"noise floor DECLARED: this run has no replicate spread, so the inert-step "
                        f"guard uses the manifest's spread_host_pct_declared = {spread_used}%")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

        below_noise = False
        # A rise needs no noise floor ("the peak went up" is not a claim about noise), so it is
        # checked separately and stays live on one-replicate systems where spread_used is empty.
        if (isinstance(d_used, int) and d_used > 0 and peak_used
                and spread_used in ("", None) and r["cross_pool"] != "yes"):
            note = (f"{pool_word} peak ROSE {d_used} kB, {100.0 * d_used / peak_used:.2f}% of the "
                    f"peak. This system runs one replicate so there is no spread to judge it "
                    f"against, but a rise is not a reduction on any reading")
            r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            below_noise = True
        if isinstance(d_used, int) and spread_used not in ("", None) and peak_used:
            move_pct = 100.0 * abs(d_used) / peak_used
            if move_pct < float(spread_used):
                what = "drop" if d_used < 0 else ("rise" if d_used > 0 else "change")
                note = (f"{pool_word} {what} {abs(d_used)} kB is {move_pct:.2f}% of the peak, BELOW this "
                        f"step's own run-to-run spread of {spread_used}% over {n_runs} runs: "
                        f"the peak did not move, so this is not shown to be a step")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
                below_noise = True
            elif d_used > 0 and r["cross_pool"] != "yes":
                # Measurably worse, and not a cross_pool placement (which would be the one defence).
                note = (f"{pool_word} peak ROSE {d_used} kB, {move_pct:.2f}% of the peak, against this step's "
                        f"own spread of {spread_used}%: measurably worse than its predecessor, which "
                        f"is not a reduction on any reading")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
                below_noise = True

        # wall_s and frozen_ms are the median replicate's (one coherent pair with the published peak).
        w = wall_seconds(sd, med_dir)
        r["wall_s"] = w if w is not None else ""
        fz = frozen_ms(sd, med_dir)
        r["frozen_ms"] = fz or ""
        # But the runtime comparison runs on the median of every replicate's own corrected wall
        # (corrected_walls()), which can differ; both are published side by side.
        cw = corrected_walls(sd)
        w_corr = statistics.median(cw) if cw else None
        if prev_w and w_corr:
            r["runtime_delta_pct"] = round(100.0 * (w_corr - prev_w) / prev_w, 1)
        if fz and w:
            note = (f"frozen {fz} ms of {w} s wall ({100.0 * fz / 1000.0 / w:.2f}%) for atomic "
                    f"captures, both from the median replicate")
            if w_corr:
                note += (f"; runtime compared on the median of the replicates' own corrected walls, "
                         f"{w_corr:.1f} s over n={len(cw)}")
            r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

        # Free/paid/marginal, decided against this step's own wall spread, not a threshold.
        _, wsp = wall_spread_pct(sd)
        r["wall_spread_pct"] = wsp
        rd = r["runtime_delta_pct"]
        if isinstance(rd, float) and wsp not in ("", None):
            sp = float(wsp)
            # Three outcomes: paid (> 2x spread, unambiguous), free (<= spread), marginal (between).
            # Marginal is deliberately excluded from the ordering check so no violation rests on a
            # coin flip (a two-outcome version made SHAFT-CPU s1 paid at +1.1% vs 0.971% spread and
            # failed the line). The factor of 2 is a uniform statistical margin, not a per-system threshold.
            if rd > 2.0 * sp:
                r["cost_type"] = "paid"
            elif rd <= sp:
                r["cost_type"] = "free"
                # A wide spread makes "free" mean "not shown to cost anything", not "shown to cost
                # nothing" (llama o3 ran 421/461/647 s), so a wide band adds an evidence sentence.
                # 10% gates only that sentence, never the classification, so it may be a round number.
                if sp > 10.0:
                    note = (f"runtime {rd:+.1f}% is inside this step's own wall spread of "
                            f"{sp:.2f}%, but that spread is WIDE: a cost up to {sp:.0f}% would "
                            f"also have landed inside it. `free` here means the measurement did "
                            f"not show a cost, which is different from showing there is none; the case for "
                            f"free rests on the step's mechanism")
                    r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            else:
                r["cost_type"] = "marginal"
                note = (f"runtime {rd:+.1f}% against this step's own wall spread of {sp:.2f}%: "
                        f"above the noise but not clearly outside it, so free-vs-paid is not "
                        f"decided by the measurement")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
        elif r["conformant"] != "baseline" and isinstance(rd, float):
            # One replicate has no spread to measure a class from, and an empty cost_type is an
            # unstated claim (rule 6's free-before-paid check would skip it). So the step DECLARES
            # its class with an argument, marked declared-not-measured; else it is "undetermined".
            dcl = declared_cost_class(sd)
            if dcl:
                r["cost_type"] = dcl
                note = (f"cost class '{dcl}' is DECLARED rather than measured: this system runs one "
                        f"replicate, so there is no wall spread to judge the {rd:+.1f}% runtime "
                        f"change against. The argument for it is in this step's levers.env and "
                        f"carries the whole weight the measurement would otherwise carry")
            else:
                r["cost_type"] = "undetermined"
                note = (f"runtime moved {rd:+.1f}% and this system runs one replicate, so there is "
                        f"no measured spread to judge it against and free-vs-paid is UNDETERMINED. "
                        f"Declare it in levers.env as `# cost_class: free` or `paid` with the "
                        f"argument; an empty class would be an unstated claim")
            r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

        for pool, peak in (("vram", vram), ("host", host_dec)):
            tot = sum(objs[pool].values())
            # Free ranking-trust check: allocated total vs resident peak. Close = the profile is
            # effectively measuring resident memory and its ordering is sound; far apart = much of
            # what it ranks was never touched and the top object may not dominate the peak (LLAMA d3: 2.72x).
            if peak and tot:
                if (tot / 1024) > peak:
                    # Allocated exceeds resident (untouched pages): coverage is not answerable in
                    # these units, and reporting 0.0 would falsely claim perfect attribution.
                    r[f"unattributed_{pool}"] = ""
                    note = (f"attributed {int(tot/1024)} kB exceeds the {peak} kB resident peak "
                            f"({tot/1024/peak:.2f}x): allocated-vs-resident rather than coverage")
                    r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
                else:
                    r[f"unattributed_{pool}"] = round(1 - (tot / 1024) / peak, 4)
                # Identified, not merely accounted for: anon:/instrument: are located but unnamed
                # (never a target), so excluding them stops named_frac reading as near-perfect.
                named = sum(v for k, v in objs[pool].items()
                            if not k.startswith(("anon:", "instrument:")))
                r[f"named_frac_{pool}"] = round((named / 1024) / peak, 4)
                # As sizes (kB), host only: named-heap sum = lower bracket on what a step could
                # target, anon total = upper bracket on what might still hide (replaces retired W_live).
                if pool == "host":
                    heap_kb = round(sum(v for k, v in objs[pool].items()
                                        if k.startswith("heap:")) / 1024)
                    anon_kb = round(sum(v for k, v in objs[pool].items()
                                        if k.startswith("anon:")) / 1024)
                    r["sum_heap"] = heap_kb or ""
                    r["peak_anon"] = anon_kb or ""

        # pool_required = larger-W pool at the PREVIOUS peak (where this step was chosen);
        # pool_attacked = what the step declares. Conformance is the two agreeing.
        r["pool_attacked"] = declared_pool(sd)
        r["pool_required"] = prev_required or ""
        r["object_attacked"] = declared_object(sd)
        r["object_required"] = prev_top.get((r["pool_attacked"] or "").strip().lower(), prev_object)

        # Largest attackable object in the attacked pool (file:/anon:/instrument: excluded).
        # Excluding instrument: stops it winning by default on an all-anon system (ARION nominated a
        # 136 kB pmtable in a 600 GB all-anon peak); an empty target is the honest output there.
        pool = r["pool_attacked"] or r["pool_required"] or \
            ("vram" if (r["w_vram"] or 0) >= (r["w_host"] or 0) else "host")
        grouped = group_sites(objs.get(pool) or {})
        chosen = {k: v for k, v in grouped.items()
                  if not k.startswith(("file:", "anon:", "instrument:"))}
        if chosen:
            name, size = max(chosen.items(), key=lambda kv: kv[1])
            peak_kb = vram if pool == "vram" else host
            r["object_targeted"] = name
            r["object_size"] = size // 1024
            # Merge-hazard warning: if the peak fell but the attacked object's row fell by less than
            # half the drop, the row may merge more than one object (a generic container constructor
            # merges callers, as twice on BumbleBee). Evidence, not a failure (freed neighbours count).
            att = (r.get("object_attacked") or "").strip()
            dh_ = r.get("delta_host")
            if (att and isinstance(dh_, int) and dh_ < 0 and prev_grouped.get(pool)
                    and pool == (r.get("pool_attacked") or "").strip().lower()):
                prev_size = None
                for k, v in prev_grouped[pool].items():
                    if att in k:
                        prev_size = v
                        break
                if prev_size is not None:
                    cur_size = 0
                    for k, v in grouped.items():
                        if att in k:
                            cur_size = v
                            break
                    moved_kb = (prev_size - cur_size) // 1024
                    if moved_kb < -dh_ // 2:
                        note = (f"the peak fell {-dh_} kB but the attacked object's row fell only "
                                f"{moved_kb} kB: the rest of the drop landed outside the declared "
                                f"object, or the row merges more than one object (the b3/b8 merge "
                                f"hazard); read the row's callers before treating it as exhausted")
                        r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            tot_attr = sum(chosen.values()) + sum(
                v for k, v in grouped.items() if k.startswith(("file:", "anon:")))
            if tot_attr:
                r["object_frac_of_attributed"] = round(size / tot_attr, 4)
            r["object_source"] = objs_meta.get(pool, {}).get(
                "source", "pagemap+smaps" if name.startswith("heap:") else "torch_state")

            # Coverage, published not enforced: targeted object vs total anon: at the same peak.
            # Below 1.0 the peak has stopped being made of nameable objects (a system property: on
            # the LLAMA dealer the remainder is glibc's arenas, which seven levers failed to reduce),
            # so it is not a build failure. The hard rule stands: anon: can never be a step's target.
            unnamed = sum(v for k, v in grouped.items() if k.startswith("anon:"))
            if unnamed and size:
                r["object_vs_unnamed"] = round(size / unnamed, 2)
                if size < unnamed:
                    note = (f"targeted object {size/1e6:.1f} MB is smaller than the "
                            f"{unnamed/1e6:.1f} MB of located-but-unnamed memory at this peak: "
                            f"past this point the peak is not made of nameable objects")
                    r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

        # Publish the below-peak gap for both halves: sample_rss_kb (resident table) must sit at the
        # peak; snapshot_rss_kb (smaps libraries, mostly stable) may lag and still give a usable term.
        for pl, peak in (("vram", vram), ("host", host_dec)):
            meta = objs_meta.get(pl, {})
            snap = meta.get("snapshot_rss_kb")
            if snap and peak and 100.0 * (peak - snap) / peak >= 1.0:
                note = (f"{pl} smaps snapshot at {snap} kB, "
                        f"{100.0 * (peak - snap) / peak:.1f}% below the {peak} kB peak: the "
                        f"library term is still usable, the snapshot does not describe the peak")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            srs = meta.get("sample_rss_kb")
            if srs and peak:
                gap = 100.0 * (peak - srs) / peak
                if gap >= MISALIGN_FAIL_PCT:
                    # A table this far below the peak describes a different moment; pairing it with
                    # the peak's smaps gives a residual from two instants (once: an attribution run at
                    # 900808 kB vs its clean run's 1465928 kB whose residual exceeded its own peak).
                    r["object_targeted"] = ""
                    r["object_size"] = ""
                    r["object_frac_of_attributed"] = ""
                    r["object_source"] = "MISALIGNED"
                if gap >= 1.0:
                    note = (f"{pl} objects sampled at {srs} kB, {gap:.1f}% below the "
                            f"{peak} kB peak")
                    r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            if objs_meta.get(pl, {}).get("drops"):
                note = (f"{pl} recorder dropped {objs_meta[pl]['drops']} allocations: "
                        f"the decomposition is incomplete by an unknown amount")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

        # Output-equality gate (where the system has one), checked against the BASELINE.
        cb, cr = read_transcript(sd, med_dir)
        r["comm_bytes"] = cb if cb is not None else ""
        r["comm_rounds"] = cr if cr is not None else ""
        gate = correctness_gate(sd, targets.get("gate_tier"), targets.get("gate_tolerance"),
                                targets.get("gate_tolerance_abs"), med_dir)
        if gate == "NONDETERMINISTIC":
            # Disagreeing replicates = no gate. This literal string used to compare equal to itself
            # (baseline vs step both NONDETERMINISTIC -> "pass"); now it fails the build.
            r["correctness_gate"] = "NONDETERMINISTIC"
            note = ("this step's own replicates disagree, so the system does not reproduce and no "
                    "comparison against the baseline can mean anything")
            r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
        elif gate:
            if base_gate is None:
                base_gate = gate
                r["correctness_gate"] = "baseline"
            else:
                ok, why = gate_equal(gate, base_gate, targets.get("gate_tier"),
                                     targets.get("gate_tolerance"),
                                     targets.get("gate_tolerance_abs"))
                r["correctness_gate"] = "pass" if ok else "FAIL"
                # T1s: byte-identical like T1 but on a STRUCTURAL observable (e.g. the modulus-chain
                # trace), for systems where no value observable reproduces at any tolerance.
                if ok and targets.get("gate_tier") == "T1s":
                    note = ("STRUCTURAL gate (T1s): the observable shows the step left the "
                            "structure of the computation unchanged; it cannot see a change "
                            "confined to values, and on this system no value observable can")
                    r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
                if not ok:
                    note = ("output differs from the baseline: this step changed the computation, "
                            "not only its memory, so its peak is not comparable")
                    if why:
                        note = f"{note} ({why})"
                    r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

                    # Phase B: a moved-output step is admitted only if it asked in advance and both
                    # conditions hold (same transcript bytes AND a plaintext equivalence test);
                    # neither alone suffices. Anything short stays FAIL.
                    ptest, ptest_ok = declared_plaintext_test(sd)
                    if declared_evidence(sd) == "equivalent":
                        why_not = []
                        if cb is None and prev_comm_bytes is None and targets.get("transcript_none"):
                            note_t = ("TRANSCRIPT VACUOUS: the manifest declares `transcript_none` "
                                      f"({targets['transcript_none']}); a single process exchanges "
                                      "no bytes, so 'the same bytes' holds by construction and "
                                      "phase B rests on the plaintext test alone")
                            r["evidence"] = f"{r['evidence']}; {note_t}" if r["evidence"] else note_t
                        elif cb is None or prev_comm_bytes is None:
                            why_not.append("no TRANSCRIPT line in this step or its predecessor, so "
                                           "the claim 'the same bytes' has nothing behind it")
                        elif cb != prev_comm_bytes:
                            _art_b, _art_f = declared_transcript_artefact(sd)
                            _delta = prev_comm_bytes - cb
                            _art_path = os.path.join(sd, _art_f) if _art_f else None
                            if _art_b is None:
                                why_not.append(f"the protocol exchanged {cb} bytes against the "
                                               f"predecessor's {prev_comm_bytes}: that is other "
                                               f"protocol work rather than a different message frame")
                            elif _delta <= 0:
                                why_not.append(f"declared a transcript artefact but the step ADDS "
                                               f"{-_delta} bytes; only a removal is admissible")
                            elif _art_b != _delta:
                                why_not.append(f"declared a transcript artefact of {_art_b} bytes "
                                               f"but the measured difference is {_delta}")
                            elif not (_art_path and os.path.exists(_art_path)):
                                why_not.append(f"declared a transcript artefact but its evidence "
                                               f"file '{_art_f}' is not in the step directory")
                            else:
                                note_a = (f"TRANSCRIPT ARTEFACT ADMITTED: this step removes "
                                          f"{_delta} bytes of the predecessor's {prev_comm_bytes}, "
                                          f"declared in advance and located in '{_art_f}'. Phase "
                                          f"B's 'same bytes' is relaxed HERE ONLY, for a removal "
                                          f"that is shown to predate the line; a step that ADDS "
                                          f"traffic is never admitted")
                                r["evidence"] = (f"{r['evidence']}; {note_a}"
                                                 if r["evidence"] else note_a)
                        if not ptest:
                            why_not.append("no `# plaintext_equivalence:` declared")
                        elif not ptest_ok:
                            why_not.append(f"the declared plaintext equivalence test '{ptest}' "
                                           f"is not in the step directory")
                        if why_not:
                            note = "phase B claimed and REFUSED: " + "; ".join(why_not)
                        else:
                            r["correctness_gate"] = "equivalent"
                            dr = (cr - prev_comm_rounds) if (cr is not None
                                                            and prev_comm_rounds is not None) else ""
                            note = (f"phase B: the output moved, and the protocol exchanged the "
                                    f"SAME {cb} bytes in {dr:+d} rounds" if dr != "" else
                                    (f"phase B: the output moved, and the protocol exchanged the "
                                     f"same {cb} bytes" if cb is not None else
                                     "phase B: the output moved, and no bytes are exchanged"))
                            note += (f"; equivalence rests on that plus the plaintext test "
                                     f"'{ptest}', and this row claims the protocol did the same "
                                     f"work, NOT that the output is identical")
                        r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

        if not (r["peak_host"] or r["peak_vram"]):
            # Unmeasured step: neither conformant nor not; "yes" would let a missing run pass.
            r["conformant"] = "no data"
        elif not r["pool_required"]:
            r["conformant"] = "baseline"
        elif not r["pool_attacked"]:
            r["conformant"] = "undeclared"
            note = "no `pool_attacked` in levers.env, so the pool choice cannot be verified"
            r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
        elif r["pool_attacked"] == r["pool_required"]:
            r["conformant"] = "yes"
        elif r["pool_required"] in declared_pool_skips(sd):
            # Larger-W pool deliberately skipped and declared (pool choice is a convention, not
            # forced); the skip must be dispositioned with a mechanism in off-path.md.
            r["conformant"] = "yes-skipped"
            note = (f"pool '{r['pool_required']}' had the larger W and was declared skipped; its "
                    f"disposition and mechanism must be in off-path.md")
            r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
        else:
            r["conformant"] = "no"
            note = (f"attacked={r['pool_attacked']} but required={r['pool_required']} "
                    f"(larger W at the previous peak)")
            r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

        # Applied after the pool verdict (where it used to be overwritten): a right-pool right-object
        # step is still not a step if the peak did not move beyond its own noise.
        if below_noise and r["conformant"] in ("yes", "yes-skipped"):
            _pm = declared_phase_effect(sd)
            _admit = False
            if _pm and prev_sd:
                _hp, _hc = marker_hwm_kb(prev_sd, _pm), marker_hwm_kb(sd, _pm)
                if _hp and _hc:
                    _fell = 100.0 * (_hp - _hc) / _hp
                    _floor = float(spread_used) if spread_used not in ("", None) else 0.0
                    if _fell >= _floor and _floor > 0:
                        _admit = True
                        note = (f"PLATEAU STEP: the global peak stayed inside the spread, but the declared "
                                f"phase (high-water at `{_pm}`) fell {_hp} -> {_hc} kB ({_fell:.2f}%, "
                                f"above the {_floor}% floor): the step removed its object from its "
                                f"phase and the peak moved to the next phase of the plateau. Admitted "
                                f"on that declaration; the plateau is cleared only by the following "
                                f"steps together (PROCEDURE, plateau)")
                    else:
                        note = (f"declared `phase_effect: {_pm}` but the high-water at that marker "
                                f"fell only {_fell:.2f}% ({_hp} -> {_hc} kB), inside the floor")
                    r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            if _admit:
                r["conformant"] = "yes-plateau"
            else:
                r["conformant"] = "no"
                inert_ids.add(sid)

        # The other half of the ordering rule: the largest OBJECT at the previous peak.
        if r["object_required"] and r["conformant"] in ("yes", "yes-skipped", "yes-plateau", "no"):
            _skips = declared_skips(sd)
            # A target must be an object: anon: (located, unnamed), file: (library text) and
            # instrument: (recorder table) can be ranked but never attacked (PUMA s1_arena_cap caps
            # glibc arenas: peak -227.4 MB, anon: -245.8, named heap -6.7). The narrow exception is a
            # retention aggregate (go:/sealpool:) whose size is a documented function of a declared
            # parameter (GOGC, or SEAL's pool holding 42% of MOAI-CPU's peak), admitted only when the
            # step's mechanism IS that knob (`runtime_knob: yes`, see arion/steps/a2_gogc50).
            _knob = declared_runtime_knob(sd)
            if r["object_attacked"].startswith(RETENTION_PREFIXES) and _knob:
                r["object_conformant"] = "yes-runtime-knob"
                note = (f"declared an attack on the runtime-retention aggregate "
                        f"'{r['object_attacked'][:40]}', admitted because levers.env declares "
                        f"`runtime_knob: yes`: the row's size is a documented function of a declared "
                        f"parameter and the step's mechanism IS that parameter. The disposition and "
                        f"its argument must be in the step's levers.env")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            elif r["object_attacked"].startswith(("anon:", "file:", "instrument:")
                                                 + RETENTION_PREFIXES):
                r["object_conformant"] = "no"
                note = (f"declared an attack on '{r['object_attacked'][:50]}', which is not an "
                        f"object this procedure can target: `anon:` is located but unnamed, "
                        f"`file:` is library text, `instrument:` is the recorder's own table and "
                        f"`go:` and `sealpool:` are retention aggregates (admissible only with "
                        f"`runtime_knob: yes`, see arion/steps/a2_gogc50). "
                        f"Such a row can be ranked and never attacked")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            elif not r["object_attacked"]:
                r["object_conformant"] = "undeclared"
            elif r["object_attacked"] in caller_half(r["object_required"]):
                r["object_conformant"] = "yes"
            elif identifies(r["object_attacked"],
                            prev_grouped.get((r.get("pool_attacked") or "").strip().lower()) or {}) \
                    == r["object_required"]:
                # The declared name is not in the (generic) caller half, but uniquely identifies the
                # required object in the full table -- the recording-pool case, allocator half.
                r["object_conformant"] = "yes"
            elif any(s and (s in caller_half(r["object_required"])
                            or identifies(s, prev_grouped.get((r.get("pool_attacked") or "")
                                                              .strip().lower()) or {})
                            == r["object_required"]) for s in _skips):
                # A skip may name the object by its caller half or, as a declaration may, by a
                # substring that uniquely identifies it in the full table (recording-pool tables
                # have a generic program half, so the allocator half is what discriminates).
                # Larger object declared skipped (rule is "largest with a FREE fix"); must be
                # dispositioned with a mechanism in off-path.md.
                r["object_conformant"] = "yes-skipped"
                note = (f"larger object '{r['object_required'][:60]}' declared skipped, so this "
                        f"step attacks the largest one with a free fix; the disposition and its "
                        f"mechanism must be in off-path.md")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            else:
                r["object_conformant"] = "no"
                note = (f"attacked object '{r['object_attacked']}' but the largest object at the "
                        f"previous peak was '{r['object_required'][:70]}'")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

            # Every object ranked ABOVE the attacked one must be declared, not only rank 1 (else a
            # step could skip rank 1 and attack rank 5 silently). Same attackable table; silent when
            # the attacked object is not in the previous table (a re-derived rename), where rank-1 applies.
            _pa = (r.get("pool_attacked") or "").strip().lower()
            _pg = prev_grouped.get(_pa) or {}
            _att = (r.get("object_attacked") or "").strip()
            if _att and _pg and r["object_conformant"] in ("yes", "yes-skipped"):
                _own = next((v for k, v in _pg.items()
                             if _att in caller_half(k) or _att in k), None)
                if _own is not None:
                    _undeclared = [
                        (k, v) for k, v in _pg.items()
                        if v > _own
                        and _att not in caller_half(k) and _att not in k
                        and not any(s and (s in caller_half(k) or s in k) for s in _skips)
                    ]
                    if _undeclared:
                        k, v = max(_undeclared, key=lambda kv: kv[1])
                        r["object_conformant"] = "no"
                        note = (f"attacked an object of {_own // 1048576} MB while "
                                f"{len(_undeclared)} larger object(s) at the previous peak carry "
                                f"no `object_skipped` declaration, the largest being "
                                f"'{caller_half(k)[:70]}' at {v // 1048576} MB")
                        r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note

        # The larger-W pool at THIS peak is what the next step must attack (single-pool CPU systems:
        # that pool until its floor).
        wv, wh = r["w_vram"], r["w_host"]
        if wv or wh:
            prev_required = "vram" if (wv or 0) >= (wh or 0) else "host"
        prev_object = r["object_targeted"] or prev_object
        # Record the top attackable object of BOTH pools, so a pool-switching successor is judged
        # against the table it was chosen from.
        for _pool in ("vram", "host"):
            _g = {k: v for k, v in group_sites(objs.get(_pool) or {}).items()
                  if not k.startswith(("file:", "anon:", "instrument:"))}
            if _g:
                prev_top[_pool] = max(_g.items(), key=lambda kv: kv[1])[0]
                prev_grouped[_pool] = _g

        prev_v, prev_h = vram or prev_v, host or prev_h
        if cb is not None:
            prev_comm_bytes, prev_comm_rounds = cb, cr
        # Carry the CORRECTED wall forward, so a frozen run is not compared against an unfrozen
        # predecessor's raw wall (which would read the instrument's pause as the lever's cost).
        prev_w = w_corr or prev_w
        prev_sd = sd
        rows.append(r)
        per_step[sid] = dict(objs, host_dec=host_dec, dec_run=dec_run)
    return rows, per_step, targets, inert_ids


def fmt_kb(v):
    return f"{int(v)/1048576:.2f} GiB" if v not in ("", None) else "--"


def write_steps_md(sysdir, rows, targets):
    """The human-readable waterfall. Same numbers as steps.csv, produced from the same rows.

    Both files are generated here so they cannot disagree: a hand-maintained prose table that
    drifts from the machine-readable one is exactly the discrepancy this campaign exists to avoid.
    """
    name = os.path.basename(sysdir.rstrip("/"))
    two_pool = any(r["peak_vram"] for r in rows)
    tier = targets.get("gate_tier")
    # T1s must be in this table: without it, the missing-declaration fallback string ran for the
    # three T1s systems (llama, sigma-gpu, moai-gpu), producing a self-contradictory header.
    tiers = {"T1": "byte-identical to the baseline's output",
             "T1s": "STRUCTURAL and byte-identical: the observable is not the output but the shape "
                    "of the computation that produced it, admitted here only because this system's "
                    "value observable cannot fail and only after a discrimination test showed the "
                    "observable moves when the computation does. What it does not cover is stated "
                    "per step and in MANIFEST",
             "T2": "numerically equal to the baseline's output within a declared tolerance",
             "T3": "label only: one bit, weak evidence, see MANIFEST",
             "T0": "NONE: no output check; a step here is not shown to preserve semantics"}
    # AND THE FALLBACK IS NOW LOUD, so this cannot recur quietly. Adding T1s to the table above
    # fixes the three systems that have it; a tier added tomorrow and forgotten here would print
    # the same self-contradicting sentence again. The tool knows whether the tier was declared, so
    # it says which of the two things went wrong instead of describing both as the same absence.
    if tier and tier not in tiers:
        raise SystemExit(f"{name}: MANIFEST declares gate_tier '{tier}', which this generator "
                         f"cannot describe. Add it to the tier table in steps_md() -- printing "
                         f"the undeclared-tier sentence under a declared tier is worse than "
                         f"failing here.")
    undeclared = ("the MANIFEST does not declare a gate tier, so no step on this line is shown to "
                  "preserve semantics")
    out = [f"# {name}: waterfall\n",
           "Generated by `tools/make_steps.py` from the raw logs. Every number here is also in "
           "`steps.csv`; neither is hand-entered.\n",
           f"Correctness gate: {tier or 'undeclared'}. "
           f"{tiers.get(tier, undeclared)}. "
           "The gate compares each step against the BASELINE's own output rather than against an "
           "external reference: an external check tests whether the system is correct, only this "
           "tests whether the step changed anything.\n"]
    if not two_pool:
        out.append("Single pool (host). This is the degenerate case of the two-pool protocol "
                   "rather than an exception to it.\n")
    if any(r["peak_party"] for r in rows):
        mism = [r["step_id"] for r in rows
                if r["decomposed_party"] and r["decomposed_party"] != r["peak_party"]]
        out.append("Multi-party: the host peak of each run is the maximum over the parties, and "
                   "the published peak is the median over runs of that maximum (`peak_party` in "
                   "`steps.csv` names whose it is). "
                   + (f"On {', '.join(f'`{s}`' for s in mism)} the heavier party has no usable "
                      f"object table of its own, so the decomposition is the other party's "
                      f"(`decomposed_party`): the table the row carried under the earlier "
                      f"freeze-leader convention, from the median run by the leader's peak. Its "
                      f"shares are taken against that party's own peak in that run, and the row's "
                      f"evidence names the run.\n"
                      if mism else
                      "Every row's decomposition is the published party's own table.\n"))

    hdr =["step", "lever", "peak", "delta", "W", "object targeted", "frac", "wall s",
           "evidence", "conformant"]
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        pool = "vram" if two_pool and r["pool_attacked"] != "host" else "host"
        obj = str(r["object_targeted"])
        obj = (obj[:70] + "...") if len(obj) > 70 else obj
        out.append("| " + " | ".join([
            r["step_id"],
            str(r["lever"])[:40] or "--",
            fmt_kb(r[f"peak_{pool}"]),
            fmt_kb(abs(int(r[f"delta_{pool}"]))) if r[f"delta_{pool}"] not in ("", None) else "--",
            str(r[f"w_{pool}"] or "--"),
            f"`{obj}`" if obj else "--",
            str(r["object_frac_of_attributed"] or "--"),
            str(r["wall_s"] or "--"),
            # The evidence class travels with the row (phase-B "equivalent" claims less than "exact").
            {"equivalent": "equivalent", "pass": "exact", "baseline": "--"}.get(
                r["correctness_gate"], str(r["correctness_gate"] or "--")),
            str(r["conformant"] or "--"),
        ]) + " |")

    if any(r["correctness_gate"] == "equivalent" for r in rows):
        out.append("")
        out.append("This line has phase B steps (marked `equivalent`), and they claim less than "
                   "the ones above them. Their output does NOT match the baseline's: on this "
                   "protocol a step that repartitions work draws different masks, and a fixed-point "
                   "truncation error follows the masks. What is shown instead is that the protocol "
                   "did the same work: the same bytes exchanged, from the step's own runs, plus a "
                   "plaintext test that the lever computes the same function. README.md, \"Two "
                   "phases of evidence\". Every step above the first `equivalent` row is "
                   "output-identical, so a reader who wants only the strongest claim can read the "
                   "line down to that point.")
        out.append("")

    first = next((r for r in rows if r["peak_host"] or r["peak_vram"]), None)
    last = next((r for r in reversed(rows) if r["peak_host"] or r["peak_vram"]), None)
    if first and last and first is not last:
        pool = "vram" if two_pool else "host"
        a, b = first[f"peak_{pool}"], last[f"peak_{pool}"]
        if a and b:
            out.append(f"\n{int(a)/int(b):.1f}x over {len(rows) - 1} steps "
                       f"({fmt_kb(a)} -> {fmt_kb(b)}).")
    out.append("\nSee each step's `peak-objects.md` for the decomposition its choice came from, "
               "and `off-path.md` for trades excluded from this table.\n")
    open(os.path.join(sysdir, "STEPS.md"), "w").write("\n".join(out))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sysdir = sys.argv[1]
    check = "--check" in sys.argv
    rows, per_step, targets, inert_ids = collect(sysdir)
    if not rows:
        sys.exit(f"no step directories under {sysdir}/steps/")

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SCHEMA, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    text = buf.getvalue()

    csv_path = os.path.join(sysdir, "steps.csv")
    # --check falls through to every conformance report below and skips only WRITING; it used to
    # return here, so a non-conformant row passed --check (what a reader runs) while failing a plain
    # write run (found 2026-08-13 on bolt/shaft-cpu). The reader's invocation must tell the truth.
    if check:
        cur = open(csv_path).read() if os.path.exists(csv_path) else ""
        if cur != text:
            sys.exit(f"MISMATCH: {csv_path} differs from a fresh regeneration")
        print(f"OK: {csv_path} matches regeneration")
    else:
        open(csv_path, "w").write(text)

    for r in ([] if check else rows):
        sd = os.path.join(sysdir, "steps", r["step_id"])
        with open(os.path.join(sd, "peak-objects.md"), "w") as fh:
            fh.write(f"# {r['step_id']}: object decomposition at each pool's peak\n\n")
            fh.write("Generated by `tools/make_steps.py`. The two pools peak at different "
                     "moments and are never summed; each is decomposed at its own peak.\n\n")
            for pool in ("vram", "host"):
                peak = r[f"peak_{pool}"]
                if not peak:
                    continue
                fh.write(f"## {pool.upper()}: peak {peak} kB")
                if r[f"w_{pool}"]:
                    fh.write(f", W = {r[f'w_{pool}']}")
                fh.write("\n\n")
                if pool == "host" and r["peak_party"]:
                    fh.write(f"Published peak: the maximum over the parties, here the "
                             f"{r['peak_party']}'s.")
                    if r["decomposed_party"] and r["decomposed_party"] != r["peak_party"]:
                        peak = per_step[r["step_id"]]["host_dec"]
                        fh.write(f" That party has no usable object table of its own, so the table "
                                 f"below is the {r['decomposed_party']}'s from "
                                 f"{per_step[r['step_id']]['dec_run']}, and its shares are taken "
                                 f"against that party's own peak there, {peak} kB.")
                    fh.write("\n\n")
                o = per_step[r["step_id"]].get(pool) or {}
                if not o:
                    fh.write("No decomposition recorded for this pool.\n\n")
                    continue
                # instrument: rows are excluded from the decomposition denominator (as steps.csv
                # already does; they disagreed by 9.5 MB on SIGMA-GPU s6) but still reported below,
                # since "the tool cost this much and it was taken off" is checkable evidence.
                instr_rows = {k: v for k, v in o.items() if str(k).startswith("instrument:")}
                o = {k: v for k, v in o.items() if not str(k).startswith("instrument:")}
                if not o:
                    fh.write("No decomposition recorded for this pool.\n\n")
                    continue
                tot = sum(o.values())
                fh.write(f"Attributed total {tot/1e6:.1f} MB against a {int(peak)/1048576:.2f} GiB "
                         f"resident peak.")
                if (tot / 1024) > int(peak):
                    fh.write(f" The total EXCEEDS the peak ({tot/1024/int(peak):.2f}x) because the "
                             f"allocator profiler reports allocated bytes while the peak is "
                             f"resident: pages allocated but never touched are not resident.\n\n"
                             f"Both denominators are given below. `frac of peak` is the quantity "
                             f"the procedure reasons about (how much of the peak this object "
                             f"accounts for), but it can exceed 1.0 for exactly the reason above, "
                             f"so it is an upper bound on the object's resident share. `frac of "
                             f"attributed` is what the ordering ranks on, and is unaffected by the "
                             f"allocated-vs-resident gap as long as that gap is roughly uniform "
                             f"across objects.")
                fh.write("\n\n| object | size | frac of peak | frac of attributed |\n"
                         "|---|---:|---:|---:|\n")
                for name, size in sorted(o.items(), key=lambda kv: -kv[1])[:15]:
                    fh.write(f"| `{name}` | {size/1e6:.1f} MB | "
                             f"{(size/1024)/int(peak):.3f} | {size/tot:.3f} |\n")
                # Keep the number and its display form separate: they were one variable until an
                # explanatory sentence replacing an empty unattributed crashed float() below.
                u_num = r[f'unattributed_{pool}']
                u_disp = (u_num if u_num != ""
                          else "not computable in these units (allocated exceeds resident)")
                if instr_rows:
                    fh.write("\ninstrument, EXCLUDED from the total above and already subtracted "
                             "from the peak (`" + "`, `".join(sorted(instr_rows)) + "`): "
                             + f"{sum(instr_rows.values())/1e6:.1f} MB\n")
                nf = r[f'named_frac_{pool}']
                fh.write(f"\nnamed (`heap:` + `file:`, i.e. the share of the peak whose IDENTITY "
                         f"is known and which a step could therefore target): {nf if nf != '' else '--'}\n\n")
                fh.write(f"unattributed (covered by no entry at all): {u_disp}\n\n")
                if isinstance(nf, float) and isinstance(u_num, float):
                    rest = round(1.0 - nf - u_num, 4)
                    if rest > 0.01:
                        fh.write(f"The remaining {rest:.1%} is `anon:`, memory the full scan "
                                 f"LOCATED but could not name. pagemap answers what is at an "
                                 f"address but not who put it there, so these bytes are evidence "
                                 f"about where the peak lives and are not eligible as a step's "
                                 f"target.\n\n")

    if not check:
        write_steps_md(sysdir, rows, targets)

    # Warn about a missing target only for a pool this system HAS (a CPU system has no VRAM pool).
    miss = [p for p, v in targets.items()
            if v is None and any(r[f"peak_{p}"] for r in rows)]
    if not check:
        print(f"wrote {csv_path} ({len(rows)} steps)")
    if miss:
        print(f"  WARNING: no target for {', '.join(miss)} in MANIFEST.md -- W not computable, "
              f"so the pool choice cannot be verified for those pools")

    # A non-conformant row is a build failure, but is still written so it stays inspectable.
    gated = [r for r in rows if r["correctness_gate"] in ("FAIL", "NONDETERMINISTIC")]
    if gated:
        for r in gated:
            if r["correctness_gate"] == "NONDETERMINISTIC":
                print(f"  NO GATE {r['step_id']}: the replicates disagree, so this system "
                      f"does not reproduce and no step of it is shown to preserve semantics")
            else:
                print(f"  GATE FAILED {r['step_id']}: {r['evidence']}")

    skew = [r for r in rows if r["object_source"] == "MISALIGNED"]
    if skew:
        for r in skew:
            print(f"  DECOMPOSITION NOT AT THE PEAK {r['step_id']}: {r['evidence']}")

    bad = [r for r in rows if r["conformant"] == "no" and r["step_id"] not in inert_ids]
    if bad:
        for r in bad:
            print(f"  NON-CONFORMANT POOL {r['step_id']}: {r['evidence']}")

    inert = [r for r in rows if r["step_id"] in inert_ids]
    if inert:
        for r in inert:
            print(f"  PEAK DID NOT MOVE {r['step_id']}: {r['evidence']}")

    # Rule 6: free fixes precede paid, so a free step after a paid one is an ordering violation.
    # Position-1 paid is not flagged; undetermined/marginal are inert (only measured/declared classes
    # count). A free-after-paid step may stand if it declares the paid step it required and names the
    # earlier run that refuted the same lever (see declared_prerequisite for the four conditions).
    misordered, seen_paid, excused = [], None, []
    # Position in the line (rows are already in measured order, see _step_order), not a parsed number.
    _pos = {x["step_id"]: i for i, x in enumerate(rows)}

    def _grouped_host(step_id):
        """That step's grouped, attackable host objects (median run), keyed by full name."""
        d = os.path.join(sysdir, "steps", step_id)
        reps = sorted(glob.glob(os.path.join(d, "logs", "run*")))
        objs, _t, _m = load_objects(d, "host", reps[len(reps) // 2] if reps else None)
        g = group_sites(objs)
        return {k: v for k, v in g.items()
                if not k.startswith(("file:", "anon:", "instrument:"))}

    def _enabler_ok(enab, row):
        """Is `enab` a valid `not_selectable_until:` enabler for this free-after-paid row?"""
        if enab not in _pos:
            return False, f"`not_selectable_until: {enab}` is not a step of this line"
        if rows[_pos[enab]]["cost_type"] != "paid":
            return False, f"the named enabler {enab} is not paid"
        if _pos[enab] >= _pos[row["step_id"]]:
            return False, f"the named enabler {enab} does not come EARLIER in the line"
        att = (row.get("object_attacked") or "").strip()
        if not att:
            return False, "no object_attacked to check for exposure"
        # Selectable = rule 4 would pick it, i.e. it is the LARGEST attackable object. The enabler
        # must make the attacked object the largest when it was not the largest before (on the stock
        # pool the key bank is dead at the peak, folded into retention; a tiny residual may remain,
        # so mere presence is not the test).
        enab_tbl = _grouped_host(enab)
        if not enab_tbl:
            return False, f"{enab} has no attackable object table"
        # Retention aggregates (sealpool:, go:) rank but are never attackable without a knob, and
        # objects this step declares skipped (paid-only fix, live at the instant) are passed over by
        # rule 4 itself; neither counts as "the largest attackable object" here.
        _sk = declared_skips(os.path.join(sysdir, "steps", row["step_id"]))
        _cand = {k: v for k, v in enab_tbl.items()
                 if not k.startswith(RETENTION_PREFIXES) and not any(s and s in k for s in _sk)}
        enab_top = max(_cand, key=_cand.get) if _cand else None
        if identifies(att, enab_tbl) != enab_top:
            # One paid enabler can expose a RUN of free fixes (MOAI-CPU m1 pool_threshold -> m2..m6):
            # each later one becomes the largest only after the earlier ones are applied. It is
            # still an enabler case if this row's own object check passed (largest attackable, after
            # declared skips, at ITS predecessor) and the object was not selectable before the enabler.
            if row.get("object_conformant") not in ("yes", "yes-skipped"):
                return False, (f"the attacked object is not the largest attackable object in {enab}'s "
                               f"table, and this step's own object check did not pass either")
        pred_id = rows[_pos[enab] - 1]["step_id"]        # enablers are paid, so pos >= 1
        pred_tbl = _grouped_host(pred_id)
        if pred_tbl and att in max(pred_tbl, key=pred_tbl.get):
            return False, (f"the attacked object was ALREADY the largest attackable object at "
                           f"{pred_id} (before the enabler), so it was selectable and this is a real "
                           f"free-after-paid inversion")
        return True, None

    for r in rows:
        if r["cost_type"] == "paid":
            seen_paid = r["step_id"]
        elif r["cost_type"] == "free" and seen_paid:
            sd = os.path.join(sysdir, "steps", r["step_id"])
            # Second admission: an ENABLER that made this object selectable (see declared_enabler).
            _enab = declared_enabler(sd)
            if _enab is not None:
                _ok, _ewhy = _enabler_ok(_enab, r)
                if _ok:
                    note = (f"free step ordered after the paid step {seen_paid}, ADMITTED as an "
                            f"enabler case: it declares `not_selectable_until: {_enab}`, and its "
                            f"attacked object is absent from {_enab}'s predecessor table and present "
                            f"in {_enab}'s, so the paid step exposed a free fix the method could not "
                            f"select before. Rule 6 orders free before paid because nothing bought "
                            f"may have been free earlier; this object was not selectable earlier")
                    r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
                    excused.append(r["step_id"])
                    continue
                # declared but invalid: fall through, recording why on the row
                r["evidence"] = (f"{r['evidence']}; `not_selectable_until: {_enab}` does not hold: "
                                 f"{_ewhy}") if r["evidence"] else \
                                f"`not_selectable_until: {_enab}` does not hold: {_ewhy}"
            req, refdir, pk_before, pk_after = declared_prerequisite(sd)
            why = None
            if req is None:
                why = None  # nothing declared: the violation stands
            elif req not in _pos:
                why = f"declares `requires: {req}`, which is not a step of this line"
            elif rows[_pos[req]]["cost_type"] != "paid":
                why = f"declares `requires: {req}`, but that step is not paid"
            elif _pos[req] >= _pos[r["step_id"]]:
                why = f"declares `requires: {req}`, which does not come EARLIER in the line"
            elif pk_after < pk_before * (1.0 - (targets.get("spread_host_pct_declared") or 0.0) / 100.0):
                # "did not reduce the peak" is read against the system's declared run-to-run spread
                # (PROCEDURE, rule 6 admissions; the same reading rule 5 applies to a plateau step).
                why = (f"declares `refuted_as: {refdir}` with {pk_before} -> {pk_after}, which is a "
                       f"REDUCTION beyond the declared spread "
                       f"({targets.get('spread_host_pct_declared') or 0.0}%): that run did not "
                       f"refute the lever, so it was available earlier")
            elif not os.path.isfile(os.path.join(sysdir, "off-path-runs", refdir, "RESULT.md")):
                why = (f"declares `refuted_as: {refdir}`, but "
                       f"off-path-runs/{refdir}/RESULT.md does not exist, so the two peaks it "
                       f"quotes cannot be audited")
            else:
                note = (f"free step ordered after the paid step {seen_paid}, ADMITTED: it declares "
                        f"`requires: {req}` and the same lever was measured earlier as "
                        f"off-path-runs/{refdir}, where it did NOT reduce the peak beyond the declared spread "
                        f"({pk_before} -> {pk_after}). Rule 6 orders free fixes before paid ones "
                        f"because nothing may be bought that was available for nothing; this lever "
                        f"was not available. What is NOT established is that {req} is the EARLIEST "
                        f"such step, which no column can check and which this step argues in its "
                        f"own text")
                r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
                excused.append(r["step_id"])
                continue
            note = (f"free step ordered AFTER the paid step {seen_paid}: free fixes are exhausted "
                    f"before paid ones, so this ordering is inverted")
            if why:
                note = f"{note}; its prerequisite declaration does not hold: it {why}"
            r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            misordered.append(r)
    for s in excused:
        print(f"  free-after-paid ADMITTED {s}: declared prerequisite holds (earlier refutation on "
              f"disk, or an enabler that exposed the object); see the row's evidence")
    if misordered:
        for r in misordered:
            print(f"  FREE AFTER PAID {r['step_id']}: {r['evidence']}")

    # Same shape on the evidence axis: exact (phase A) steps come before equivalent (phase B) ones,
    # or the line reached for the weaker claim while the stronger was still available.
    misphased, seen_equiv = [], None
    for r in rows:
        if r["correctness_gate"] == "equivalent":
            seen_equiv = r["step_id"]
        elif r["correctness_gate"] == "pass" and seen_equiv:
            note = (f"exact-evidence step ordered AFTER the equivalence-evidence step {seen_equiv}: "
                    f"output-identical steps are exhausted first, so this ordering is inverted")
            r["evidence"] = f"{r['evidence']}; {note}" if r["evidence"] else note
            misphased.append(r)
    if misphased:
        for r in misphased:
            print(f"  EXACT AFTER EQUIVALENT {r['step_id']}: {r['evidence']}")

    nodata = [r for r in rows if r["conformant"] == "no data"]
    if nodata:
        for r in nodata:
            print(f"  NO MEASUREMENT {r['step_id']}: the step directory exists but carries no "
                  f"peak for either pool, so this row is a placeholder rather than a result")

    # The object half of the ordering rule, held to the same standard as the pool half.
    badobj = [r for r in rows if r["object_conformant"] == "no"]
    if badobj:
        for r in badobj:
            print(f"  NON-CONFORMANT OBJECT {r['step_id']}: attacked "
                  f"'{r['object_attacked']}' but the largest object at the previous peak was "
                  f"'{r['object_required'][:60]}'")

    # A lever step with no decomposition at all (four BumbleBee reductions once exited 0 with "No
    # decomposition recorded"). Counted separately from MISALIGNED: the reader's next action differs
    # (derive the run's tables from its own artifacts, vs re-measure).
    notable = [r for r in rows if r["object_attacked"] and not r["object_targeted"]
               and r["object_source"] != "MISALIGNED" and r["conformant"] != "no data"]
    if notable:
        for r in notable:
            print(f"  NO DECOMPOSITION {r['step_id']}: the step declares an attack on "
                  f"'{r['object_attacked']}' but carries no object table for its pool, so there "
                  f"is no evidence of which object fell")

    # A gate that cannot fail. A baseline hides one (it defines the reference rather than being
    # compared), so a degenerate observable passes silently; every paradigm has produced one.
    degen = []
    for r in rows:
        d = gate_degenerate(os.path.join(sysdir, "steps", r["step_id"]))
        if d:
            degen.append((r, d))
    if degen:
        for r, (n, v) in degen:
            print(f"  DEGENERATE GATE {r['step_id']}: the observable is {n} copies of {v!r}, so "
                  f"it reproduces perfectly and distinguishes nothing; a step compared against "
                  f"this reference cannot fail the gate, whatever it changes")

    # Replication summary, printed always: the noise floor each row was judged against, in the build log.
    reped = [r for r in rows if r.get("n_runs")]
    if reped:
        print("  replication (median published, extremes carried):")
        for r in reped:
            print(f"    {r['step_id']:<22} n={r['n_runs']}  "
                  f"min={r['peak_host_min']}  max={r['peak_host_max']}  "
                  f"spread={r['spread_host_pct']}%")
        thin = [r for r in reped if r["n_runs"] < 3]
        if thin:
            print(f"  NOTE: {len(thin)} step(s) have fewer than three runs; their spread is not "
                  f"established and every claim of a 'free' change on them rests on one number")

    if (gated or bad or inert or badobj or skew or nodata or notable or misordered
        or misphased or degen):
        sys.exit(f"{len(gated)} step(s) failed the correctness gate, {len(bad)} attacked the "
                 f"wrong pool, {len(badobj)} attacked the wrong object, {len(inert)} did not move "
                 f"the peak, {len(skew)} carry a decomposition taken too far from their own peak, "
                 f"{len(nodata)} carry no measurement at all, {len(notable)} declare an attack "
                 f"but carry no object table for the attacked pool, {len(degen)} carry a gate "
                 f"observable that cannot fail, {len(misordered)} put a free "
                 f"step after a paid one, and {len(misphased)} put an output-identical step after "
                 f"one that only shows equivalent protocol work; such steps are not publishable "
                 f"as waterfall steps")


if __name__ == "__main__":
    main()
