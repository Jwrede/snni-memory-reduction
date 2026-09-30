#!/usr/bin/env python3
"""Resolves runtime addresses to function names against the /proc/<pid>/maps snapshot; used by
resolve_resident.py. Name without file:line (symbol table only) = nearest preceding symbol, kept
as a hint; the address stays authoritative.

Known one-page error where p_vaddr != p_offset (BumbleBee). Compensated in five derives (bumblebee,
arion, bolt, puma, moai-cpu: maps rewritten load-base-relative), not in four (shark, shaft-cpu,
shaft-gpu, moai-gpu). Do not fix here alone: the five would be corrected twice.
"""
import os
import shlex
import struct
import subprocess
import sys
from collections import defaultdict

_SYMTAB_CACHE = {}


def _sized_funcs(lib, exec_prefix):
    """Sorted [(value, size, mangled_name)] of every sized STT_FUNC symbol in `lib`.

    Read from the ELF directly, not readelf/nm, because some measured images ship no binutils (PUMA
    has no readelf). Both SYMTAB and DYNSYM are read (a stripped-not-static lib keeps sized dynsyms).
    """
    if lib in _SYMTAB_CACHE:
        return _SYMTAB_CACHE[lib]
    out = []
    try:
        if exec_prefix:
            blob = subprocess.run(shlex.split(exec_prefix) + ["cat", lib],
                                  capture_output=True, timeout=300).stdout
        else:
            with open(lib, "rb") as fh:
                blob = fh.read()
        if blob[:4] == b"\x7fELF" and blob[4] == 2:
            e_shoff, = struct.unpack_from("<Q", blob, 0x28)
            e_shentsize, e_shnum = struct.unpack_from("<HH", blob, 0x3a)
            secs = []
            for i in range(e_shnum):
                o = e_shoff + i * e_shentsize
                _n, typ, _f, _a, off, size, link, _i, _al, entsize = \
                    struct.unpack_from("<IIQQQQIIQQ", blob, o)
                secs.append((typ, off, size, link, entsize))
            for typ, off, size, link, entsize in secs:
                if typ not in (2, 11) or not entsize:      # SHT_SYMTAB, SHT_DYNSYM
                    continue
                stroff = secs[link][1]
                for i in range(size // entsize):
                    o = off + i * entsize
                    st_name, st_info, _o, _sh, st_value, st_size = \
                        struct.unpack_from("<IBBHQQ", blob, o)
                    if (st_info & 0xf) != 2 or not st_value or not st_size:
                        continue
                    ns = stroff + st_name
                    end = blob.index(b"\0", ns)
                    out.append((st_value, st_size, blob[ns:end].decode("utf8", "replace")))
            out.sort()
    except Exception:
        out = []
    _SYMTAB_CACHE[lib] = out
    return out


def _exact_name(lib, off, mangled, exec_prefix):
    """True when `off` lies INSIDE the sized symbol whose name addr2line already reported.

    THE PROBLEM THIS DECIDES. A function name with no `file:line` may come from debug info whose
    line table is absent, in which case it is EXACT, or from the symbol table, in which case it is
    the nearest PRECEDING symbol and therefore a guess. The two are indistinguishable from
    addr2line's output alone, and this tool used to assume the second and throw the name away.
    Measured 2026-08-13, that assumption discarded exact names covering 99.8% of MOAI-CPU's peak,
    94.1% of PUMA's, 43.4% of SHAFT-GPU's and 14.7% of BOLT's, including an object BOLT's `s2`
    declares that it attacks.

    IT IS DECIDABLE, because ELF symbols carry a SIZE. Inside `[st_value, st_value + st_size)` the
    name is the containing function; past the end it is the nearest preceding one and stays a
    guess. Both halves were measured before this was written: `python3.10+0x13cc27` lands inside
    `PyBytes_FromStringAndSize` and is kept, `python3.10+0x28728a` lands 0x12da past the end of
    `PyInit_select` and is refused.

    TWO INDEPENDENT SOURCES MUST AGREE, which is what makes this safe to run everywhere. The
    offsets handed to addr2line are file offsets and symbol values are virtual addresses; where a
    segment has `p_vaddr != p_offset` those differ by a page, and this file's own header documents
    that five derives pre-compensate for it and four do not. Requiring the symbol found by
    containment to be the SAME symbol addr2line named means a mismatched convention cannot
    manufacture a name: it produces a disagreement and the address stays an offset. The comparison
    is on the MANGLED name so that no demangler is needed inside the image.
    """
    if not mangled or mangled in ("??", ""):
        return False
    syms = _sized_funcs(lib, exec_prefix)
    if not syms:
        return False
    lo, hi = 0, len(syms)
    while lo < hi:                       # rightmost symbol with value <= off
        mid = (lo + hi) // 2
        if syms[mid][0] <= off:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return False
    v, sz, name = syms[lo - 1]
    return off < v + sz and name == mangled


_A2L_SEEN = {}


def _have_addr2line(exec_prefix):
    """Is there an addr2line where the caller wants it run, and SAY SO if there is not.

    MEASURED 2026-08-13 ON PUMA, and it is the reason this exists. Its derive runs the symboliser
    with `--exec "apptainer exec <sif>"`, i.e. INSIDE the measured image, and that image ships no
    binutils at all: only `cat`. Every addr2line invocation therefore failed, the exception was
    swallowed, and every address in the published table fell back to `libspu.so+0x356fb8a`. The
    line then stopped because "77% of the peak is one unnameable row", which was recorded as a
    property of the SYSTEM and was a property of the IMAGE.

    `derive_host_objects.sh` documents the identical shape from an earlier month, when the exec
    prefix named podman on a cluster that has none. Twice is enough: a missing symboliser is now
    reported once per prefix instead of being inferred from a table full of offsets.
    """
    if exec_prefix in _A2L_SEEN:
        return _A2L_SEEN[exec_prefix]
    cmd = (shlex.split(exec_prefix) if exec_prefix else []) + ["addr2line", "--version"]
    try:
        ok = subprocess.run(cmd, capture_output=True, timeout=120).returncode == 0
    except Exception:
        ok = False
    if not ok:
        sys.stderr.write(
            "symbolise: NO addr2line where the symboliser was told to run it "
            f"({exec_prefix or 'this host'}); falling back to the ELF symbol table, which needs "
            "no binutils. Names below come from sized FUNC symbols by containment.\n")
        sys.stderr.flush()
    _A2L_SEEN[exec_prefix] = ok
    return ok


def _demangle(names):
    """Itanium-ABI demangling, run LOCALLY: it is a string transform and needs no binary.

    The image holding the library may have no `c++filt` (PUMA's does not) while the host running
    the derive does. Falling back to the mangled name is still better than dropping it: `_ZN8pybind11...`
    identifies a function uniquely, an offset does not.
    """
    want = sorted({n for n in names if n.startswith("_Z")})
    if not want:
        return {}
    try:
        r = subprocess.run(["c++filt"], input="\n".join(want), capture_output=True,
                           text=True, timeout=120)
        return {a: b.strip() for a, b in zip(want, r.stdout.splitlines()) if b.strip()}
    except Exception:
        return {}


def _symtab_name(lib, off, exec_prefix):
    """The sized FUNC symbol CONTAINING `off`, or None. Used only when addr2line is unavailable.

    Containment is the same test `_exact_name` applies, minus the corroboration by addr2line,
    because here there is no addr2line to corroborate with. What it still refuses is an address
    past the end of every symbol, which is the nearest-preceding guess Falle 11 is about. What it
    cannot detect alone is a systematic offset-convention error, so the provenance is announced by
    `_have_addr2line` rather than left for a reader to infer from the names.
    """
    syms = _sized_funcs(lib, exec_prefix)
    if not syms:
        return None
    lo, hi = 0, len(syms)
    while lo < hi:
        mid = (lo + hi) // 2
        if syms[mid][0] <= off:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return None
    v, sz, name = syms[lo - 1]
    return name if off < v + sz else None


def resolve(addrs, maps, exec_prefix, inline=False, src_out=None):
    """addr -> 'func (file:line)', by batching addr2line per mapped object.

    `src_out`, when a dict is passed, is FILLED with the full source path per address --
    `/root/moai/include/test/test_full_scheme.hpp` rather than the basename this function returns.
    It exists so a caller can ask which frames belong to the MEASURED PROGRAM and which to a
    library, which is a question only the directory can answer: `evaluator.h` and
    `test_full_scheme.hpp` are both plausible basenames and only one of them is the program. The
    return value is untouched, so every existing caller is byte-identical.

    `inline=True` asks addr2line for the INLINE CHAIN (-i) and returns the OUTERMOST frame, i.e.
    the function the allocation was inlined INTO, rather than the wrapper it was inlined FROM.
    Default off, so every existing caller keeps byte-identical names.

    WHY IT EXISTS. On MOAI-GPU the single largest device entry is
    `cuda_wrapper.cuh:26:void check<cudaError>(...)` at 49.3% of the peak: the error-check wrapper
    every CUDA call is routed through. The recorder already distinguishes those call sites -- they
    are distinct addresses -- but addr2line without -i names them all after the wrapper they were
    inlined from, so a decomposition that is perfectly correct cannot discriminate. FINDINGS lists
    this as the cheapest of the three keys that swallowed a peak, and the debug info needed to
    resolve it is already in the binary.

    ONE ADDRESS PER CALL when inline is on, because -i emits a variable number of lines per
    address and batching would misalign them. There are tens of sites, not thousands.
    """
    by_lib = defaultdict(list)
    for a in addrs:
        for start, end, fileoff, lib in maps:
            if start <= a < end and lib.startswith("/"):
                by_lib[lib].append((a, a - start + fileoff))
                break
    out = {}
    prefix = shlex.split(exec_prefix) if exec_prefix else []
    # Asked ONCE per prefix, before any work: a missing addr2line used to turn every address into
    # an offset without a word, which is how PUMA published a table that was 94% unnameable.
    have_a2l = _have_addr2line(exec_prefix)
    fallback = {}

    if inline:
        for lib, pairs in by_lib.items():
            base = os.path.basename(lib)
            for a, o in pairs:
                if not have_a2l:
                    n = _symtab_name(lib, o, exec_prefix)
                    if n:
                        fallback[a] = n
                        out[a] = n
                    else:
                        out[a] = f"{base}+0x{o:x}"
                    continue
                try:
                    res = subprocess.run(prefix + ["addr2line", "-f", "-C", "-i", "-e", lib,
                                                   f"0x{o:x}"],
                                         capture_output=True, text=True, timeout=60)
                    lines = [l.strip() for l in res.stdout.splitlines()]
                except Exception:
                    lines = []
                # addr2line -f -i prints func/loc pairs from the INNERMOST frame outwards, so the
                # last pair is the function the code was inlined into. That is the one a step can
                # be aimed at; the inner ones are the wrapper.
                pairs_fl = [(lines[i], lines[i + 1]) for i in range(0, len(lines) - 1, 2)]
                named = [(fn, lc) for fn, lc in pairs_fl
                         if fn not in ("??", "") and lc not in ("??:0", "??:?", "??", "")]
                if named:
                    fn, lc = named[-1]
                    out[a] = f"{os.path.basename(lc)}:{fn}"
                    if src_out is not None:
                        src_out[a] = lc.rsplit(":", 1)[0] if ":" in lc else lc
                else:
                    # No frame carried a file:line: before a raw offset, check whether addr2line's
                    # outermost name is exact by containment (this branch once dropped such names,
                    # leaving MOAI-CPU's 28 h m1 run 99.8% unnamed while addr2line said Ciphertext::resize).
                    out[a] = f"{base}+0x{o:x}"
                    outer = [fn for fn, _lc in pairs_fl if fn not in ("??", "")]
                    if outer:
                        try:
                            raw = subprocess.run(prefix + ["addr2line", "-f", "-i", "-e", lib,
                                                           f"0x{o:x}"],
                                                 capture_output=True, text=True, timeout=60)
                            mang = [l.strip() for l in raw.stdout.splitlines()[::2]]
                        except Exception:
                            mang = []
                        if mang and _exact_name(lib, o, mang[-1], exec_prefix):
                            out[a] = outer[-1]
        _apply_demangle(out, fallback)
        return out

    for lib, pairs in by_lib.items():
        base = os.path.basename(lib)
        if not have_a2l:
            for a, o in pairs:
                n = _symtab_name(lib, o, exec_prefix)
                if n:
                    fallback[a] = n
                    out[a] = n
                else:
                    out[a] = f"{base}+0x{o:x}"
            continue
        offs = [f"0x{o:x}" for _a, o in pairs]
        try:
            res = subprocess.run(prefix + ["addr2line", "-f", "-C", "-e", lib] + offs,
                                 capture_output=True, text=True, timeout=300)
            lines = res.stdout.splitlines()
        except Exception:
            lines = []
        base = os.path.basename(lib)
        # The SAME batch again without -C, so the containment test below can compare mangled name
        # against mangled ELF symbol and need no demangler inside the image. One extra addr2line
        # per library, not per address.
        mangled = {}
        try:
            raw = subprocess.run(prefix + ["addr2line", "-f", "-e", lib] + offs,
                                 capture_output=True, text=True, timeout=300)
            rl = raw.stdout.splitlines()
            for i, (a, _o) in enumerate(pairs):
                if 2 * i < len(rl):
                    mangled[a] = rl[2 * i].strip()
        except Exception:
            mangled = {}
        for i, (a, o) in enumerate(pairs):
            func = lines[2 * i].strip() if 2 * i < len(lines) else "??"
            loc = lines[2 * i + 1].strip() if 2 * i + 1 < len(lines) else "??"
            # A name without a file:line is the nearest exported symbol, not the allocator, and reads
            # like a bogus call stack on a stripped binary; demoted to a hint, address authoritative.
            if loc in ("??:0", "??:?", "??", ""):
                # It may still be exact with only its line table missing; decide by containment.
                out[a] = f"{base}+0x{o:x}"
                if func not in ("??", ""):
                    if mangled.get(a) and _exact_name(lib, o, mangled[a], exec_prefix):
                        out[a] = func
                    else:
                        out[a] += f"(near {func})"
            else:
                out[a] = f"{os.path.basename(loc)}:{func}"
                if src_out is not None:
                    src_out[a] = loc.rsplit(":", 1)[0] if ":" in loc else loc
    _apply_demangle(out, fallback)
    return out


def _apply_demangle(out, fallback):
    """Demangle the symbol-table names in place, once, at the end.

    Batched deliberately: one `c++filt` for the whole table rather than one per address, and only
    over the names that actually came from the symbol table, so nothing addr2line already
    demangled with `-C` is touched twice.
    """
    if not fallback:
        return
    dem = _demangle(fallback.values())
    for a, mangled in fallback.items():
        if mangled in dem:
            out[a] = dem[mangled]

