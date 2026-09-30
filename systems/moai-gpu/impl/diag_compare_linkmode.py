#!/usr/bin/env python3
"""Link-mode decision from the six diagnostic runs (diag_compare_linkmode.py <diag-runs-dir>).

Question: does dynamic cudart linking change the measured peak? Rule fixed in advance:
within = max(spread(S), spread(D)), between = |median(D)-median(S)|/median(S);
between<=within INDISTINGUISHABLE (pmrec-style recorder), <=2*within MARGINAL (more replicates),
else DISTINGUISHABLE (CUPTI route). Voting peaks: reserved (primary), 100 ms poll (secondary); they
must agree. Pool live high-water reported, not voting (zero noise).
"""
import os
import re
import sys

VARIANTS = ["build-static", "build-shared"]
REPS = [0, 1, 2]


def read_run(d):
    """Return (guard_failures, reserved_kb, alloc_kb, polled_mib, wall_s, nmark)."""
    fails = []
    peak = os.path.join(d, "PEAK.txt")
    if not os.path.isdir(d):
        return ["run directory missing"], None, None, None, None, 0

    wall = None
    p = os.path.join(d, "wall_seconds.txt")
    if os.path.exists(p):
        wall = int(open(p).read().strip() or 0)

    nmark = 0
    p = os.path.join(d, "markers.log")
    if os.path.exists(p):
        nmark = sum(1 for ln in open(p) if ln.startswith("MEM|"))

    reserved = alloc = None
    p = os.path.join(d, "all_markers.txt")
    if os.path.exists(p):
        for ln in open(p):
            m = re.search(r"peak_reserved_kb=(\d+)", ln)
            if m:
                reserved = int(m.group(1))
            m = re.search(r"peak_alloc_kb=(\d+)", ln)
            if m:
                alloc = int(m.group(1))

    polled = None
    if os.path.exists(peak):
        m = re.search(r"poll_vram_peak_MiB=(\d+)", open(peak).read())
        if m:
            polled = int(m.group(1))

    # The three guards, restated here so the analysis does not trust the job's own verdict.
    if nmark < 16:
        fails.append(f"{nmark}/16 phase markers")
    if reserved is None:
        fails.append("no GPUPEAK line")
    if wall is None or wall < 600:
        fails.append(f"wall {wall}s < 600s")
    rc = os.path.join(d, "exit_status.txt")
    if os.path.exists(rc) and open(rc).read().strip() not in ("0",):
        fails.append(f"exit {open(rc).read().strip()}")
    return fails, reserved, alloc, polled, wall, nmark


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def spread(xs):
    m = median(xs)
    return (max(xs) - min(xs)) / m if m else float("nan")


def verdict(s_vals, d_vals, label, unit):
    within = max(spread(s_vals), spread(d_vals))
    ms, md = median(s_vals), median(d_vals)
    between = abs(md - ms) / ms if ms else float("nan")
    print(f"\n--- {label} ---")
    print(f"  static  {s_vals}  median {ms} {unit}  spread {spread(s_vals)*100:.2f}%")
    print(f"  dynamic {d_vals}  median {md} {unit}  spread {spread(d_vals)*100:.2f}%")
    print(f"  within  (max of the two spreads) {within*100:.2f}%")
    print(f"  between (medians)                {between*100:.2f}%")
    if between <= within:
        v = "INDISTINGUISHABLE"
    elif between <= 2 * within:
        v = "MARGINAL"
    else:
        v = "DISTINGUISHABLE"
    print(f"  => {v}")
    return v


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = sys.argv[1].rstrip("/")

    data = {}
    bad = False
    print("=== guards, per run ===")
    for v in VARIANTS:
        data[v] = {"reserved": [], "polled": [], "alloc": []}
        for r in REPS:
            d = os.path.join(root, v, f"run{r}")
            fails, reserved, alloc, polled, wall, nmark = read_run(d)
            state = "ok" if not fails else "REJECTED: " + "; ".join(fails)
            print(f"  {v} run{r}: wall={wall}s markers={nmark} "
                  f"reserved={reserved} alloc={alloc} polled={polled} MiB -> {state}")
            if fails:
                bad = True
            else:
                data[v]["reserved"].append(reserved)
                data[v]["polled"].append(polled)
                data[v]["alloc"].append(alloc)

    if bad:
        print("\nAT LEAST ONE RUN FAILED ITS GUARDS. No verdict: a comparison over a set that "
              "includes a run of the wrong program is exactly the mistake this file exists to "
              "prevent. Fix and re-run rather than dropping the bad replicate quietly.")
        sys.exit(1)

    v1 = verdict(data["build-static"]["reserved"], data["build-shared"]["reserved"],
                 "device peak, pool reserved high-water (primary, exact)", "kB")
    v2 = verdict(data["build-static"]["polled"], data["build-shared"]["polled"],
                 "device peak, 100 ms poll (secondary)", "MiB")

    # Reported, does not vote. See the header for why.
    a_s, a_d = data["build-static"]["alloc"], data["build-shared"]["alloc"]
    print("\n--- pool LIVE high-water (diagnostic, does not vote) ---")
    print(f"  static  {a_s}")
    print(f"  dynamic {a_d}")
    if len(set(a_s)) == 1 and len(set(a_d)) == 1:
        ds, dd = a_s[0], a_d[0]
        if ds == dd:
            print("  identical in every run of both builds: the live set is the same program, and "
                  "the whole run-to-run band is allocator fragmentation.")
        else:
            print(f"  deterministic within each build but DIFFERENT between them: "
                  f"{dd - ds:+d} kB ({(dd - ds) / ds * 100:+.3f}%). The two builds allocate "
                  "differently; read the verdict above knowing that.")
    else:
        print("  not deterministic within a build, unlike the static set measured on 2026-07-31. "
              "That is itself new and should be understood before the verdict is used.")

    print("\n=== decision ===")
    if v1 != v2:
        print(f"  The two mechanisms disagree ({v1} vs {v2}). That disagreement is the result: "
              "neither number is used until it is understood.")
        sys.exit(2)
    print(f"  Both mechanisms agree: {v1}")
    if v1 == "INDISTINGUISHABLE":
        print("  -> link cudart dynamically, declare it, build the recorder as pmrec.")
    elif v1 == "MARGINAL":
        print("  -> three more replicates per variant before choosing. Do not decide on this.")
    else:
        print("  -> the link mode moves the peak. CUPTI injection route instead.")


if __name__ == "__main__":
    main()
