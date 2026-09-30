#!/usr/bin/env python3
"""Test of symbolise.py: exact names kept, nearest-symbol guesses refused.

Criterion: ELF symbol size. Address inside [st_value, st_value+st_size): containing function;
past the end: offset only. Builds its own fixture (needs gcc):

    with_line      compiled -g            -> file:func, unchanged behaviour
    no_line        compiled -g0, kept     -> a sized FUNC symbol with no line table
    past_the_end   an address after the last symbol's end -> must stay an offset

Run: python3 tools/test_symbolise.py
"""
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import symbolise  # noqa: E402

SRC_LINE = r"""
#include <stdlib.h>
void *with_line(unsigned long n) { return malloc(n); }
"""

SRC_NOLINE = r"""
#include <stdlib.h>
void *no_line(unsigned long n) { return calloc(n, 1); }
int main(void) { return 0; }
"""


def elf_funcs(path):
    with open(path, "rb") as fh:
        blob = fh.read()
    e_shoff, = struct.unpack_from("<Q", blob, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", blob, 0x3a)
    secs = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        secs.append(struct.unpack_from("<IIQQQQIIQQ", blob, o))
    out = {}
    for sec in secs:
        typ, off, size, link, entsize = sec[1], sec[4], sec[5], sec[6], sec[9]
        if typ not in (2, 11) or not entsize:
            continue
        stroff = secs[link][4]
        for i in range(size // entsize):
            o = off + i * entsize
            st_name, st_info, _o, _sh, st_value, st_size = struct.unpack_from("<IBBHQQ", blob, o)
            if (st_info & 0xf) != 2 or not st_value or not st_size:
                continue
            ns = stroff + st_name
            out[blob[ns:blob.index(b"\0", ns)].decode()] = (st_value, st_size)
    return out


def main():
    if not shutil.which("gcc") or not shutil.which("addr2line"):
        print("SKIP: needs gcc and addr2line")
        return 0
    tmp = tempfile.mkdtemp(prefix="symtest-")
    try:
        a = os.path.join(tmp, "a.c")
        b = os.path.join(tmp, "b.c")
        binp = os.path.join(tmp, "prog")
        open(a, "w").write(SRC_LINE)
        open(b, "w").write(SRC_NOLINE)
        # One object WITH debug info and one WITHOUT, linked together: the situation that produces
        # a name with no line table inside a binary that does have .debug_info.
        subprocess.run(["gcc", "-g", "-O0", "-c", a, "-o", os.path.join(tmp, "a.o")], check=True)
        subprocess.run(["gcc", "-g0", "-O0", "-c", b, "-o", os.path.join(tmp, "b.o")], check=True)
        subprocess.run(["gcc", os.path.join(tmp, "a.o"), os.path.join(tmp, "b.o"),
                        "-o", binp, "-no-pie"], check=True)

        funcs = elf_funcs(binp)
        for need in ("with_line", "no_line"):
            if need not in funcs:
                print(f"SKIP: toolchain produced no sized symbol for {need}")
                return 0

        wl_v, wl_sz = funcs["with_line"]
        nl_v, nl_sz = funcs["no_line"]
        last_end = max(v + s for v, s in funcs.values())

        # -no-pie, so runtime address == file offset: keeps this test about the naming rule, not
        # the load-base arithmetic the header documents.
        addrs = [wl_v + 1, nl_v + 1, last_end + 0x40]
        maps = [(0, last_end + 0x1000, 0, binp)]

        failures = []
        for inline in (False, True):
            got = symbolise.resolve(addrs, maps, "", inline=inline)
            tag = "inline" if inline else "batch"

            g = got.get(wl_v + 1, "")
            if not re.match(r"^a\.c:\d+:with_line$", g):
                failures.append(f"[{tag}] with_line: debug-info name changed, got {g!r}")

            g = got.get(nl_v + 1, "")
            if g != "no_line":
                failures.append(f"[{tag}] no_line: exact name not kept, got {g!r}")

            g = got.get(last_end + 0x40, "")
            if "no_line" in g or re.match(r"^[a-z_]+$", g):
                failures.append(f"[{tag}] past_the_end: a guess was published as a name, got {g!r}")
            elif "+0x" not in g:
                failures.append(f"[{tag}] past_the_end: expected an offset, got {g!r}")

        # Size test itself, independent of addr2line: one byte before the end is inside, one byte
        # after is not, so this can't pass by coincidence on adjacent symbols.
        if not symbolise._exact_name(binp, nl_v + nl_sz - 1, "no_line", ""):
            failures.append("[containment] last byte of the symbol reported as outside")
        if symbolise._exact_name(binp, nl_v + nl_sz, "no_line", ""):
            failures.append("[containment] first byte PAST the symbol reported as inside")
        # A name that disagrees with the symbol found by containment must never be accepted: the
        # guard that makes a page-off offset convention fail closed rather than invent one.
        if symbolise._exact_name(binp, nl_v + 1, "some_other_function", ""):
            failures.append("[containment] a disagreeing name was accepted")

        if failures:
            for f in failures:
                print("FAIL: " + f)
            return 1
        print("PASS: a name with debug info is unchanged")
        print("PASS: a name without a line table is kept when the address is INSIDE its symbol")
        print("PASS: an address past the last symbol stays an offset, in both branches")
        print("PASS: containment is exclusive at the end, and a disagreeing name is refused")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
