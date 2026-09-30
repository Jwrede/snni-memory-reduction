#!/usr/bin/env python3
"""Extracts the inline patch heredocs of a build script into files; the script is not modified.

    extract_inline_patches.py <build script> <output dir> [--check]

Target: SIGMA-GPU's build script (levers as `python3 - "$TARGET" <<'TAG' ... TAG` heredocs).
Naming: index prefix (application order) + the block's own `print("  patched: ...")` text.
--check: re-extracts and compares byte for byte.
"""
import os
import re
import sys

USAGE = "extract_inline_patches.py <build script> <output dir> [--check]"

# `python3 - "<args>" <<'TAG'` ... newline TAG. The quoted tag is what makes the body literal, so a
# block written with an unquoted tag would be shell-expanded and is NOT the same text; those are
# refused rather than extracted wrongly.
OPEN_RE = re.compile(r"^python3 - (?P<args>.*?)<<'(?P<tag>[A-Za-z0-9_]+)'\s*$")


def slug(text):
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", s)[:48]


def blocks(path):
    """Every inline heredoc block, in the order the script applies them."""
    lines = open(path, errors="replace").read().splitlines(keepends=True)
    out, i, index = [], 0, 0
    while i < len(lines):
        m = OPEN_RE.match(lines[i].rstrip("\n"))
        if not m:
            i += 1
            continue
        tag, start = m.group("tag"), i + 1
        end = start
        while end < len(lines) and lines[end].rstrip("\n") != tag:
            end += 1
        if end >= len(lines):
            sys.exit(f"FAILED: heredoc '{tag}' opened at line {i+1} is never closed")
        body = "".join(lines[start:end])
        p = re.search(r'print\(\s*"\s*patched:\s*(.+?)["\(]', body)
        if not p:
            sys.exit(
                f"FAILED: the block at line {i+1} has no `print(\"  patched: ...\")` line, so it "
                f"cannot name itself and this tool will not name it for it"
            )
        index += 1
        out.append((index, slug(p.group(1)), m.group("args").strip(), body))
        i = end + 1
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv[1:]
    if len(args) != 2:
        sys.exit(USAGE)
    script, outdir = args
    if not os.path.isfile(script):
        sys.exit(f"FAILED: no build script at {script}")

    found = blocks(script)
    if not found:
        sys.exit(f"FAILED: {script} contains no inline patch blocks; nothing to extract")

    bad, n = [], 0
    for index, name, targets, body in found:
        fname = f"inline_{index:02d}_{name}.py"
        dest = os.path.join(outdir, fname)
        header = (
            f"# EXTRACTED, NOT AUTHORED. This is block {index} of\n"
            f"# `{os.path.basename(script)}`, lifted out verbatim by "
            f"`tools/extract_inline_patches.py`.\n"
            f"#\n"
            f"# The script is unchanged and is still what produced the published binaries; this "
            f"file exists\n"
            f"# so that one step's lever can be picked up on its own. The two are compared byte "
            f"for byte by\n"
            f"# `--check`, so neither can drift from the other.\n"
            f"#\n"
            f"# Applied in the script as: python3 - {targets} <<...\n"
            f"# Take it as: python3 {fname} <the same target path>\n\n"
        )
        content = header + body
        if check:
            if not os.path.isfile(dest):
                bad.append(f"{fname}: missing")
            elif open(dest, errors="replace").read() != content:
                bad.append(f"{fname}: differs from the block in the script")
        else:
            os.makedirs(outdir, exist_ok=True)
            open(dest, "w").write(content)
            print(f"  wrote {fname}  (applies to {targets})")
        n += 1

    if check:
        if bad:
            for b in bad:
                print("  " + b)
            sys.exit(f"FAILED: {len(bad)} of {n} extracted patch(es) do not match the script")
        print(f"OK: all {n} extracted patches match the blocks in {os.path.basename(script)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
