#!/usr/bin/env python3
"""Peak anatomy figure: per step, RSS trace with the peak instant and the object decomposition there.

    plot_peak_anatomy.py systems/<sys> [out.png]

Shared memory axis; object colours fixed across steps; `anon:` entries hatched.
"""
import json
import os
import re
import sys
import glob
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TOP_NAMED = 5          # named objects drawn individually per step; the rest is collected
LEGEND_MAX = 12        # objects named in the legend; chosen by how often they LEAD a step
COLS = 4               # step pairs per row

# NOT `tab20`: its light/dark pairs of one hue land directly on top of each other when colours
# are ranked by size while the bar stacks by size. Okabe-Ito (CVD-safe) extended with Paul Tol's
# muted set, ordered so neighbours in the stack are neighbours in hue.
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # purple-pink
    "#56B4E9",  # sky
    "#D55E00",  # vermillion
    "#117733",  # dark green
    "#AA4499",  # magenta
    "#88CCEE",  # pale cyan
    "#999933",  # olive
    "#882255",  # wine
    "#44AA99",  # teal
    "#661100",  # brown
    "#F0E442",  # yellow
]


def published(sysdir, pool):
    """Step order AND the published peak per step, both from steps.csv. The waterfall shows the
    MEDIAN replicate's peak; the composition comes from a possibly different run (the one whose
    object table was derived), so taking the peak from the table would print a number that
    appears nowhere else (measured: 122.50x instead of SHARK's published 124.40x)."""
    import csv
    out = []
    for r in csv.DictReader(open(os.path.join(sysdir, "steps.csv"))):
        try:
            pk = float(r[f"peak_{pool}"])
        except (KeyError, ValueError):
            pk = None
        out.append((r["step_id"], pk, r.get("object_attacked", "")))
    return out


def declared_skips(sysdir, step):
    """`# object_skipped:` lines of a step's levers.env: the larger objects it passed over,
    declared rather than inferred (PROCEDURE requires this)."""
    f = os.path.join(sysdir, "steps", step, "levers.env")
    out = []
    if os.path.exists(f):
        for ln in open(f, errors="replace"):
            m = re.match(r"#\s*object_skipped:\s*(.+)", ln.strip())
            if m:
                out.append(m.group(1).strip())
    return out


def _key_words(s):
    """The identifying words of an object name, for matching a declaration against a table row.
    A recording-pool key is two frames, `object <- allocator`, and a declaration may name either
    side (BumbleBee names the object, MOAI's recording pool names the allocator). Split on `<-` and
    take the function words of every frame. The file:line prefix, argument list and template of each
    frame are stripped per frame, so a later frame's file:line cannot swallow an earlier frame's
    function name (which the previous single-pass strip did, leaving only the allocator)."""
    s = re.sub(r"^(heap|file|anon|instrument):", "", s)
    words = []
    for frame in s.split("<-"):
        f = re.sub(r"^\s*.*?:\d+[^:]*:", "", frame)  # leading file:line[ (discriminator N)] prefix
        f = re.sub(r"\(.*", "", f)                   # argument lists differ from the declaration
        f = re.sub(r"<[^<>]*>", "", f)
        words += [w for w in re.split(r"[^A-Za-z0-9_]+", f) if len(w) > 2]
    return words


def match_row(decl, rows):
    """The table row a declaration refers to, or None. Longest word-overlap wins."""
    if not decl:
        return None
    want = _key_words(decl)
    if not want:
        return None
    best, best_score = None, 0
    # Only heap rows: `file:` entries are loaded library text, not objects, and a declaration
    # naming a library could otherwise bind to its mapping.
    for k in [k for k in rows if not k.startswith("file:")]:
        have = set(_key_words(k))
        score = sum(1 for w in want if w in have)
        if score > best_score:
            best, best_score = k, score
    return best if best_score >= max(1, len(want) // 2) else None


def load_step(sysdir, step, pool="host"):
    """(peak_kb, trace, objects) for the run whose peak is the median, or None. `pool` selects the
    decomposition drawn; the two pools are never mixed into one figure, since their maxima don't
    coincide (a system with two pools gets two figures)."""
    runs = []
    for od in sorted(glob.glob(os.path.join(sysdir, "steps", step, "logs", "run*"))):
        oj = os.path.join(od, f"objects_{pool}.jsonl")
        if not os.path.exists(oj):
            continue
        try:
            objs = json.load(open(oj))
        except Exception:
            continue
        runs.append((objs.get("rss_kb") or 0, od, objs))
    if not runs:
        return None
    runs.sort(key=lambda r: r[0])
    peak_kb, rundir, objs = runs[len(runs) // 2]

    # Trace of the party whose maximum the table describes, to avoid pairing one party's timeline
    # with another's objects. Only the host pool has a timeline (no device-memory equivalent), so
    # a VRAM figure carries no "where" panel rather than drawing a mismatched trace.
    if pool != "host":
        return peak_kb, None, objs
    best, best_gap = None, None
    for vm in glob.glob(os.path.join(rundir, "poll_*", "vmrss.log")):
        pts = []
        for ln in open(vm, errors="replace"):
            if ln.startswith("#"):
                continue
            f = ln.split()
            if len(f) >= 2:
                try:
                    pts.append((int(f[0]), int(f[1])))
                except ValueError:
                    pass
        if not pts:
            continue
        gap = abs(max(p[1] for p in pts) - peak_kb)
        if best_gap is None or gap < best_gap:
            best, best_gap = pts, gap
    return peak_kb, best, objs


def group_by_function(objects):
    """Collapse call sites of ONE function into one row, matching README.md's ranking: a
    symboliser resolves each allocation to a file:line, so one logical object allocated from
    several lines would otherwise arrive as several small rows. The grouping has a known limit
    (BumbleBee: two genuinely different objects merge into one key at depth 1)."""
    out = {}
    for k, v in objects.items():
        if k.startswith(("anon:", "file:", "instrument:")):
            out[k] = out.get(k, 0) + v
            continue
        m = re.match(r"^(heap:)(?:.*?:\d+(?:\s*\(discriminator \d+\))?:)?(.*)$", k)
        key = (m.group(1) + m.group(2)) if m else k
        out[key] = out.get(key, 0) + v
    return out


def shorten(name):
    n = re.sub(r"^(heap|file|anon|instrument):", "", name)
    n = re.sub(r"\(.*?\)", "()", n)
    n = re.sub(r"<[^<>]*>", "<>", n)
    if " <- " in n:
        a, b = n.split(" <- ", 1)
        n = a.strip().split("/")[-1] + " <- " + b.strip().split("/")[-1]
    n = n.split("/")[-1]
    return n[:52] + ("..." if len(n) > 52 else "")


def build_palette(sysdir, steps, pool="host"):
    """One colour per object across the whole figure, so a row can be followed between steps."""
    # An object earns a colour by LEADING somewhere, not by being large somewhere: a row that
    # never nears the top of any step would crowd the legend for no benefit.
    lead = {}
    for st in steps:
        d = load_step(sysdir, st, pool)
        if not d:
            continue
        named = sorted(((k, v) for k, v in group_by_function(d[2].get("objects", {})).items()
                        if not k.startswith("anon:")), key=lambda x: -x[1])
        for rank, (k, v) in enumerate(named[:3]):
            lead[k] = max(lead.get(k, 0), v)
    ranked = [k for k, _ in sorted(lead.items(), key=lambda x: -x[1])][:LEGEND_MAX]
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(ranked)}


def main():
    sysdir = sys.argv[1].rstrip("/")
    pool = "host"
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    for a in sys.argv[2:]:
        if a.startswith("--pool="):
            pool = a.split("=", 1)[1]
    name = os.path.basename(sysdir)
    out = args[0] if args else f"{name}-peak-anatomy-{pool}.png"

    pub = published(sysdir, pool)
    data = [(st, load_step(sysdir, st, pool), pk, att) for st, pk, att in pub]
    data = [(st, d, pk, att) for st, d, pk, att in data if d]
    if not data:
        sys.exit(f"no measured steps for pool {pool} under {sysdir}")
    palette = build_palette(sysdir, [s for s, _, _, _ in data], pool)
    n = len(data)

    # Three panels at three scales: the descent spans orders of magnitude on some systems (e.g.
    # SHARK 53.8 GB -> 0.5 GB), so a shared linear axis would hide the late steps' composition.
    # The waterfall keeps absolute scale; each step's trace is scaled to its own peak (only WHERE
    # the max sits matters there); each composition is normalised to 100%, with the absolute
    # value printed above it.
    # No empty frame where there is no data: drawing a 0..1 axis next to a VRAM decomposition
    # reads as a crash. Where no step has a trace, the pair collapses to the composition alone.
    has_trace = any(d[1] for _, d, _, _ in data)
    per = 2 if has_trace else 1
    rows = (n + COLS - 1) // COLS
    fig = plt.figure(figsize=((3.0 if has_trace else 1.5) * COLS + 2.2, 3.1 * rows + 2.4))
    gs = fig.add_gridspec(rows + 1, per * COLS,
                          height_ratios=[1.15] + [1] * rows,
                          width_ratios=([3, 1] * COLS) if has_trace else None,
                          hspace=0.95, wspace=0.45)

    axw = fig.add_subplot(gs[0, :])
    # published peak where steps.csv has one, else the table's own
    peaks = [(pk if pk else d[0]) / 1024.0 for _, d, pk, _ in data]
    axw.bar(range(n), peaks, color="0.72", edgecolor="0.35", linewidth=0.6, width=min(0.62, 0.28 + 0.09 * n))
    for i, v in enumerate(peaks):
        lbl = f"{v:,.0f}"
        if i:
            lbl += f"\n{(peaks[i]/peaks[i-1]-1)*100:+.0f}%"
        axw.text(i, v, lbl, ha="center", va="bottom", fontsize=7)
    axw.set_xticks(range(n))
    axw.set_xticklabels([s for s, _, _, _ in data], fontsize=7, rotation=18, ha="right")
    axw.set_ylabel(f"peak {pool}, MB", fontsize=8)
    # Log scale, only here: a 100x+ descent (122x on SHARK) flattens to a line on a linear axis.
    # Stacked bars below stay linear, since added heights must not be logarithmic.
    if max(peaks) / max(min(peaks), 1e-9) > 12:
        axw.set_yscale("log")
        axw.set_ylim(min(peaks) * 0.45, max(peaks) * 3.2)
    else:
        axw.set_ylim(0, max(peaks) * 1.25)
    axw.tick_params(labelsize=7)
    for sp in ("top", "right"):
        axw.spines[sp].set_visible(False)
    axw.set_title(f"{peaks[0]:,.0f} -> {peaks[-1]:,.0f} MB  =  {peaks[0]/peaks[-1]:.1f}x",
                  fontsize=8, loc="left", color="0.3")

    legend = OrderedDict()
    for i, (st, (peak_kb, trace, objs), pub_kb, _att) in enumerate(data):
        r, c = i // COLS, i % COLS
        axt = fig.add_subplot(gs[r + 1, per * c]) if has_trace else None
        axb = fig.add_subplot(gs[r + 1, per * c + (1 if has_trace else 0)])

        # --- WHERE: the trace, on its own scale
        pct = None
        if trace and axt is not None:
            t0 = trace[0][0]
            span = max(1, trace[-1][0] - t0)
            xs = [(q[0] - t0) / span * 100 for q in trace]
            ys = [q[1] / 1024.0 for q in trace]
            axt.fill_between(xs, ys, color="0.87", linewidth=0)
            axt.plot(xs, ys, color="0.55", linewidth=0.8)
            pi = max(range(len(ys)), key=lambda j: ys[j])
            pct = xs[pi]
            axt.axvline(pct, color="crimson", linewidth=1.1, zorder=5)
            axt.plot([pct], [ys[pi]], "o", color="crimson", ms=4, zorder=6)
            axt.set_ylim(0, max(ys) * 1.22)
        if axt is not None:
            axt.set_xlim(0, 100)
            axt.set_xlabel("% of the run", fontsize=7)
            axt.tick_params(labelsize=6)
            for sp in ("top", "right"):
                axt.spines[sp].set_visible(False)
        shown = (pub_kb if pub_kb else peak_kb) / 1024.0
        where = f", peak at {pct:.0f}%" if pct is not None else ""
        (axt or axb).set_title(f"{st}\n{shown:,.0f} MB{where}", fontsize=8, loc="left")

        # --- WHAT: the composition at that instant, as shares of the peak
        o = group_by_function(objs.get("objects", {}))
        named = sorted(((k, v) for k, v in o.items() if not k.startswith("anon:")),
                       key=lambda x: -x[1])

        # What the NEXT step does with this peak: the waterfall chooses each step from its
        # predecessor's table, so the useful annotation is the next step's attacked object and
        # its declared skips, not what this step itself did.
        nxt_att = nxt_skips = None
        if i + 1 < len(data):
            nxt_st, _, _, nxt_att_decl = data[i + 1]
            nxt_att = match_row(nxt_att_decl, dict(named))
            nxt_skips = [match_row(d, dict(named)) for d in declared_skips(sysdir, nxt_st)]
            nxt_skips = [x for x in nxt_skips if x and x != nxt_att]
        anon = sum(v for k, v in o.items() if k.startswith("anon:"))
        tot = sum(o.values()) or 1
        bottom = 0.0
        marked = False
        for k, v in named[:TOP_NAMED]:
            h = v / tot * 100
            is_att = (k == nxt_att)
            is_skip = (k in (nxt_skips or []))
            axb.bar(0, h, bottom=bottom, width=0.85, color=palette.get(k, "0.62"),
                    edgecolor=("crimson" if is_att else ("0.15" if is_skip else "white")),
                    linewidth=(2.2 if is_att else (1.8 if is_skip else 1.0)),
                    linestyle=("--" if is_skip else "-"), zorder=(4 if (is_att or is_skip) else 2))
            if h >= 9:
                axb.text(0, bottom + h / 2, f"{h:.0f}", ha="center", va="center",
                         fontsize=6, color="white")
            if is_att:
                axb.annotate("attacked by\nthe next step", xy=(0.45, bottom + h / 2),
                             xytext=(1.25, bottom + h / 2), fontsize=6, color="crimson",
                             va="center", ha="left",
                             arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0))
                marked = True
            elif is_skip:
                axb.annotate("skipped by it,\ndeclared", xy=(0.45, bottom + h / 2),
                             xytext=(1.25, bottom + h / 2), fontsize=6, color="0.3",
                             va="center", ha="left",
                             arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9,
                                             linestyle="--"))
            bottom += h
            if k in palette:
                legend.setdefault(k, palette[k])
        if nxt_att and not marked:
            axb.set_xlabel("attacked object\nnot in the top 5", fontsize=5.5,
                           color="crimson")
        rest = sum(v for _, v in named[TOP_NAMED:]) / tot * 100
        if rest > 0:
            axb.bar(0, rest, bottom=bottom, width=0.85, color="0.78",
                    edgecolor="white", linewidth=1.0)
            bottom += rest
        if anon > 0:
            axb.bar(0, anon / tot * 100, bottom=bottom, width=0.85, color="white",
                    edgecolor="0.45", hatch="///", linewidth=0.5)
        axb.set_xlim(-0.6, 2.6)
        axb.set_ylim(0, 100)
        axb.set_xticks([])
        axb.set_yticks([0, 50, 100])
        axb.tick_params(labelsize=6)
        if axt is not None:
            axb.set_title("made of", fontsize=7, color="0.35")
        # A decomposition of ONE row is a result, not a rendering failure (e.g. SIGMA-GPU's
        # device peak is a single 40 GiB reservation); say so explicitly.
        if len(named) == 1 and anon == 0:
            axb.set_xlabel("ONE object,\n100% of the peak", fontsize=5.5, color="0.3")
        for sp in ("top", "right"):
            axb.spines[sp].set_visible(False)

    handles = [Patch(facecolor=c, label=shorten(k)) for k, c in legend.items()]
    handles.append(Patch(facecolor="0.78", label="other named"))
    handles.append(Patch(facecolor="white", edgecolor="0.45", hatch="///",
                         label="unnamed (anon:), never attacked"))
    handles.append(Patch(facecolor="white", edgecolor="crimson", linewidth=2.0,
                         label="attacked by the next step"))
    handles.append(Patch(facecolor="white", edgecolor="0.25", linewidth=1.6, linestyle="--",
                         label="declared skipped by the next step"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{name} ({pool}): where the peak is and what it is made of, per reduction step",
                 fontsize=12, x=0.02, ha="left", y=0.995)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90,
                        bottom=0.12 + 0.028 * ((len(handles) + 1) // 2))
    fig.savefig(out, dpi=150)
    print(f"{out}  ({n} steps, {len(legend)} named objects in the legend)")


if __name__ == "__main__":
    main()
