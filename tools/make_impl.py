#!/usr/bin/env python3
"""Generates steps/<id>/impl/ per step: BUILD.md recipe plus copies of the named source artifacts.

    make_impl.py <system-dir> [--check]

Inputs: MANIFEST.md `impl_<key>:` block (impl_upstream + impl_commit or impl_base_image; instrument;
one impl_build_<flavour>: per form); each step's levers.env `# impl_lever: <files>` (files this step
adds) or `# impl_snapshot:`. Chain order from steps.csv. `# impl_lever: none` is explicit.
--check: regenerates into a temp tree and compares content.
"""
import filecmp
import os
import re
import shutil
import sys
import tempfile

USAGE = "make_impl.py <system-dir> [--check]"


def die(msg):
    sys.exit("FAILED: " + msg)


def read_manifest_impl(sysdir):
    """The system-wide half of the recipe: base, commit, instrument, build entry point."""
    path = os.path.join(sysdir, "MANIFEST.md")
    if not os.path.isfile(path):
        die(f"{path} is missing")
    out = {}
    for line in open(path, errors="replace"):
        m = re.match(r"\s*impl_([a-z_]+)\s*:\s*(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    # Base is one of two shapes (clone at a pinned commit, or a published base image); the file
    # declares which rather than the generator guessing.
    if not (("upstream" in out and "commit" in out) or "base_image" in out):
        die(
            f"{path} declares no base: give `impl_upstream:` with `impl_commit:`, or "
            f"`impl_base_image:`. Without it a reader cannot start, and this file is the only "
            f"place the campaign records it."
        )
    # Every build form the system has: impl_build_palma is what ran (evidenced), a portable form is
    # what a reader elsewhere needs; both are rendered and labelled.
    builds = {k[len("build_"):]: v for k, v in out.items() if k.startswith("build_")}
    if not builds:
        die(f"{path} declares no `impl_build_<flavour>:` line; a recipe without a build is not one")
    out["_builds"] = builds
    return out


def read_step_lever(stepdir, is_first):
    """The per-step half: the source artifacts THIS step adds, nothing it inherits.

    A baseline has no levers.env (it is the system before any lever), so a missing file is accepted
    only for the first step and refused elsewhere.
    """
    path = os.path.join(stepdir, "levers.env")
    if not os.path.isfile(path):
        if is_first:
            return ("lever", [])
        die(f"{path} is missing, and only the first step of a line may have no levers.env")
    txt = open(path, errors="replace").read()
    snap = re.search(r"^#\s*impl_snapshot\s*:\s*(.+?)\s*$", txt, re.M)
    lev = re.search(r"^#\s*impl_lever\s*:\s*(.+?)\s*$", txt, re.M)
    if snap and lev:
        die(f"{path} declares BOTH impl_lever and impl_snapshot; a step has one shape, not two")
    if snap:
        # A snapshot replaces the chain: some systems publish one complete file per step (SHARK's
        # bert_instrumented_*.cpp, SHAFT's module) with earlier levers already inside, so the recipe
        # must not replay them (it would apply them twice).
        v = snap.group(1).strip()
        return ("snapshot", [] if v == "none" else v.split())
    if not lev:
        die(
            f"{path} declares neither `# impl_lever:` nor `# impl_snapshot:`. Declare "
            f"`# impl_lever: none` if this step changes no source; an absent declaration and a "
            f"declaration of nothing are different claims."
        )
    v = lev.group(1).strip()
    if v == "unpublished":
        # Third case: `none` = changed no source; `unpublished` = DID change source but the change
        # was never published, only described in prose (MOAI-GPU s1_complexroots_free: a hand-added
        # destructor in fft.h/ckks.cu). Collapsing it into none would turn a gap into a claim.
        return ("unpublished", [])
    return ("lever", [] if v == "none" else v.split())


def declared_equivalence(stepdir):
    """The plaintext equivalence test a step cites, if any.

    make_steps.py resolves `# plaintext_equivalence: <path>` inside the step dir and fails the gate
    without it, so this generator must carry it (an early version deleted 14 such tests across the
    SHAFT family, flipping four systems' --check from equivalent to FAIL; the round trip caught it).
    """
    f = os.path.join(stepdir, "levers.env")
    if not os.path.isfile(f):
        return None
    m = re.search(r"plaintext_equivalence\s*[:=]\s*(\S+)", open(f, errors="replace").read())
    return m.group(1) if m else None


def dirs_equal(a, b):
    """Recursive content comparison, because a lever can be a directory of source files."""
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    if mismatch or errors:
        return False
    return all(dirs_equal(os.path.join(a, d), os.path.join(b, d)) for d in cmp.common_dirs)


def step_order(sysdir):
    """The published order, taken from steps.csv so it cannot disagree with the waterfall."""
    path = os.path.join(sysdir, "steps.csv")
    if not os.path.isfile(path):
        die(f"{path} is missing")
    import csv

    return [r["step_id"] for r in csv.DictReader(open(path))]


def build_md(sysname, meta, step_id, index, chain, unpublished):
    """The recipe for one step, in the order a reader would run it."""
    lines = []
    lines.append(f"# {sysname}: building `{step_id}`")
    lines.append("")
    lines.append(
        "Generated by `tools/make_impl.py` from this system's `MANIFEST.md`, its `steps.csv` and "
        "each step's `levers.env`. Do not edit: regenerate."
    )
    lines.append("")
    lines.append(
        f"A waterfall step is cumulative, so this is the program with {len(chain)} source "
        f"change(s) applied, in this order. It is step {index} of the published line."
    )
    lines.append("")
    lines.append("## 1. The base")
    lines.append("")
    if "base_image" in meta:
        lines.append(
            "This system is built on top of a published image rather than from a clone."
        )
        lines.append("")
        lines.append("```")
        lines.append(meta["base_image"])
        lines.append("```")
    else:
        if meta["commit"] == "unpinned":
            # Said out loud: a recipe that quietly clones HEAD reproduces a different program daily;
            # the image digest in RUN_META is then the only fixed point.
            lines.append(
                "This system's upstream is not pinned to a commit. The build clones the "
                "default branch, and neither the campaign's build scripts nor the image record "
                "which commit that was. What is fixed is the IMAGE: its sha256 is in this step's "
                "`logs/run*/RUN_META` as `image_digest`, and that is the artifact the published "
                "numbers were measured on."
            )
            lines.append("")
            lines.append("```")
            lines.append(f"git clone {meta['upstream']} src      # HEAD, unpinned")
            lines.append("cd src")
            lines.append("```")
        else:
            lines.append("```")
            lines.append(f"git clone {meta['upstream']} src")
            lines.append("cd src")
            lines.append(f"git checkout {meta['commit']}")
            lines.append("```")
    lines.append("")
    if meta.get("instrument"):
        lines.append("## 2. The campaign instrument")
        lines.append("")
        lines.append(
            "Applied identically to the baseline and to every step, so it is part of the measured "
            "system rather than a lever. The file's own header says why each part of it is not optional."
        )
        lines.append("")
        lines.append("```")
        for inst in meta["instrument"].split():
            lines.append(
                (f"python3 {inst} <source tree>") if inst.endswith(".py")
                else (f"git apply -p1 {inst}      # inside the cloned tree")
            )
        lines.append("```")
        lines.append("")
    lines.append("## 3. The levers of this step")
    lines.append("")
    if not chain:
        lines.append("None. This is the baseline: the measured system with no lever applied.")
    elif chain[0][2] == "snapshot":
        lines.append(
            "This step publishes its program as a complete source artifact rather than as a patch on its "
            "predecessor, so the earlier levers of the line are already inside it. Copy it over "
            "the tree; do not replay the steps before it, which would apply them twice."
        )
        lines.append("")
        lines.append("```")
        for owner, fname, _ in chain:
            lines.append(f"cp -r {fname} <into the tree, see ../../MANIFEST.md>")
        lines.append("```")
    else:
        lines.append("Applied in this order, on top of the base:")
        lines.append("")
        lines.append("```")
        for owner, fname, _ in chain:
            verb = "python3" if fname.endswith(".py") else "apply"
            lines.append(f"{verb} {fname} <source tree>      # from {owner}")
        lines.append("```")
    if chain:
        lines.append("")
        lines.append(
            "Each artifact is copied into this directory, so nothing outside it has to be found."
        )
    lines.append("")
    if unpublished:
        lines.append("")
        lines.append(
            "Not every source change on this line is published as an artifact. "
            + ", ".join(f"`{u}`" for u in unpublished)
            + " changed the source and the change exists only as prose in that step's "
              "`levers.env`. Reproducing this step means making those edits by hand from that "
              "description."
        )
    lines.append("")
    lines.append("## 4. Build and run")
    lines.append("")
    order = ["docker", "apptainer", "palma"]
    names = {
        "docker": "Docker, portable",
        "apptainer": "Apptainer, portable",
        "palma": "PALMA, the form this campaign actually ran",
    }
    for flavour in order + [f for f in sorted(meta["_builds"]) if f not in order]:
        if flavour not in meta["_builds"]:
            continue
        lines.append(f"{names.get(flavour, flavour)}")
        lines.append("")
        lines.append("```")
        lines.append(meta["_builds"][flavour])
        lines.append("```")
        lines.append("")
    lines.append(
        "The PALMA form is the one the published measurement came from; a portable form rebuilds "
        "the same program and cannot reproduce this campaign's machine, which `../../MANIFEST.md` "
        "pins and which every number here depends on."
    )
    lines.append("")
    lines.append(
        "`levers.env` beside this file carries the runtime settings and the declarations the build "
        "checks. `logs/` carries this step's own measurement, and `../../MANIFEST.md` the machine, "
        "the target and the gate."
    )
    lines.append("")
    return "\n".join(lines)


def generate(sysdir, outroot):
    """Write every step's impl/ under `outroot`, which is the repo itself or a temp tree."""
    sysname = os.path.basename(os.path.normpath(sysdir))
    meta = read_manifest_impl(sysdir)
    # A transfer system (SHAFT-CPU-ViT, SHAFT-GPU-ViT) runs the parent's image with a different
    # model, so its artifacts live in the parent's impl/, declared via impl_artifacts_from:.
    src_impl = os.path.join(sysdir, "impl")
    if "artifacts_from" in meta:
        src_impl = os.path.join(os.path.dirname(os.path.normpath(sysdir)), meta["artifacts_from"], "impl")
    if not os.path.isdir(src_impl):
        die(f"{src_impl} is missing; there is nothing to assemble a recipe from")

    chain = []
    unpublished = []
    written = 0
    for index, step_id in enumerate(step_order(sysdir)):
        stepdir = os.path.join(sysdir, "steps", step_id)
        if not os.path.isdir(stepdir):
            die(f"steps.csv names {step_id} but {stepdir} does not exist")
        kind, names = read_step_lever(stepdir, index == 0)
        # A declared artifact resolves against the step's own impl/ first, then the system's
        # (BumbleBee keeps each step's patches/tree in the step; most keep levers once at system level).
        local_impl = os.path.join(sysdir, "steps", step_id, "impl")
        for fname in names:
            if not (os.path.exists(os.path.join(local_impl, fname))
                    or os.path.exists(os.path.join(src_impl, fname))):
                die(
                    f"{step_id} declares `{fname}` but it is neither in {local_impl} nor in "
                    f"{src_impl}. A recipe naming a file nobody has is worse than no recipe."
                )
        unpublished.append(step_id) if kind == "unpublished" else None
        if kind == "snapshot":
            chain = [(step_id, f, "snapshot") for f in names]
        else:
            chain = [c for c in chain if c[2] != "snapshot"]
            chain += [(step_id, f, "lever") for f in names]

        out = os.path.join(outroot, "steps", step_id, "impl")
        os.makedirs(out, exist_ok=True)
        needed = {"BUILD.md"}

        # The equivalence test, and its recorded output beside it, belong to the step: the build
        # resolves them here and refuses phase B without them.
        equiv = declared_equivalence(stepdir)
        if equiv:
            base = os.path.basename(equiv)
            needed.add(base)
            for extra in (base, os.path.splitext(base)[0] + ".out"):
                have = os.path.join(sysdir, "steps", step_id, "impl", extra)
                src_alt = os.path.join(src_impl, extra)
                dst = os.path.join(out, extra)
                if os.path.isfile(have):
                    needed.add(extra)
                    if os.path.abspath(have) != os.path.abspath(dst):
                        shutil.copy2(have, dst)
                elif os.path.isfile(src_alt):
                    needed.add(extra)
                    shutil.copy2(src_alt, dst)
        for inst in meta.get("instrument", "").split():
            needed.add(inst.split("/")[0])
            inst_dst = os.path.join(out, inst)
            os.makedirs(os.path.dirname(inst_dst), exist_ok=True)
            shutil.copy2(os.path.join(src_impl, inst), inst_dst)
        for owner, fname, _kind in chain:
            needed.add(fname.split("/")[0])
            local_first = os.path.join(sysdir, "steps", owner, "impl", fname)
            src_path = (local_first if (owner == step_id and os.path.exists(local_first))
                        else os.path.join(src_impl, fname))
            dst_path = os.path.join(out, fname)
            # A declaration may name a path inside impl/ (SHARK's src/bert_instrumented_*.cpp), so
            # the parent must exist before the copy.
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if os.path.abspath(src_path) == os.path.abspath(dst_path):
                continue          # already where it belongs
            if os.path.isdir(src_path):
                shutil.rmtree(dst_path, ignore_errors=True)
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        open(os.path.join(out, "BUILD.md"), "w").write(
            build_md(sysname, meta, step_id, index, list(chain), list(unpublished))
        )
        # A file this generator did not write is a leftover from an earlier line; removed, not ignored.
        for stale in os.listdir(out):
            if stale not in needed:
                stale_path = os.path.join(out, stale)
                shutil.rmtree(stale_path) if os.path.isdir(stale_path) else os.remove(stale_path)
        written += 1
    return written


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv[1:]
    if len(args) != 1:
        sys.exit(USAGE)
    sysdir = args[0].rstrip("/")

    if not check:
        n = generate(sysdir, sysdir)
        print(f"OK: wrote impl/ for {n} step(s) of {os.path.basename(sysdir)}")
        return 0

    # --check regenerates and compares content, not mere presence: a lever renamed, a step inserted,
    # or an order changed only shows in a regeneration.
    tmp = tempfile.mkdtemp(prefix="make_impl_")
    try:
        generate(sysdir, tmp)
        bad = []
        for step_id in step_order(sysdir):
            have = os.path.join(sysdir, "steps", step_id, "impl")
            want = os.path.join(tmp, "steps", step_id, "impl")
            if not os.path.isdir(have):
                bad.append(f"{step_id}: no impl/ at all")
                continue
            names_have, names_want = sorted(os.listdir(have)), sorted(os.listdir(want))
            if names_have != names_want:
                bad.append(f"{step_id}: files differ, have {names_have} want {names_want}")
                continue
            # A declaration may name a DIRECTORY (BOLT's s1 lever is three files); filecmp.cmp on a
            # dir answers False rather than raising, so compare directories recursively.
            diff = []
            for f in names_want:
                a, b = os.path.join(have, f), os.path.join(want, f)
                if os.path.isdir(a) or os.path.isdir(b):
                    if not (os.path.isdir(a) and os.path.isdir(b) and dirs_equal(a, b)):
                        diff.append(f)
                elif not filecmp.cmp(a, b, shallow=False):
                    diff.append(f)
            if diff:
                bad.append(f"{step_id}: content differs in {diff}")
        if bad:
            for b in bad:
                print("  " + b)
            die(f"{len(bad)} step(s) carry an impl/ that does not match their declarations")
        print(f"OK: every step's impl/ regenerates from its declarations ({os.path.basename(sysdir)})")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
