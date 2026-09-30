#!/usr/bin/env python3
"""Parses a Go heap dump (`debug.WriteHeapDump`) into (address, size, site) objects.

Site = allocation stack from memory-profile records (kinds 16/17). Pass 1: kinds 16/17; pass 2:
kind-1 objects, sampled ones emitted (holding all addr->size OOM-killed at 8 GB on a 520 GiB heap).
Output: one JSON line per object {"addr", "size", "site": "<leaf frame>", "stack": [frames]};
unsampled objects get site "unsampled".
"""
import json
import sys

# Frame depth kept per object. A prior 6-frame limit truncated every one of ARION's 230,336
# `rlwe.NewElement` stacks, losing the goroutine-root frame attribution needed. Cheap here since
# the intermediate jsonl is deleted after the join.
STACK_KEEP = 24


class R:
    """A varint reader over the dump. Reads byte-by-byte deliberately: a chunked-read rewrite
    measured 0.8x (Python-level bookkeeping cost more than it saved) and was reverted. The pass
    is CPU-bound in this loop at ~168,000 objects/s, so a 570 GB dump takes about three hours."""

    def __init__(self, fh):
        self.fh = fh

    def tell(self):
        return self.fh.tell()

    def read(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError
        return b

    def uvarint(self):
        shift = 0
        val = 0
        while True:
            b = self.fh.read(1)
            if not b:
                raise EOFError
            c = b[0]
            val |= (c & 0x7F) << shift
            if not c & 0x80:
                return val
            shift += 7

    def string(self):
        n = self.uvarint()
        return self.read(n)

    def skip_string(self):
        n = self.uvarint()
        self.fh.seek(n, 1)
        return n

    def fieldlist(self):
        while True:
            kind = self.uvarint()
            if kind == 0:
                return
            self.uvarint()  # offset


def walk(dump, on_object, on_sample, on_profile):
    """One pass over the dump, calling back only for the record kinds the caller wants. Every
    kind is still parsed (needed to know its shape); only which ones are kept differs, which is
    what lets a pass fit in memory."""
    fh = open(dump, "rb")
    hdr = fh.readline()
    if not hdr.startswith(b"go1.7 heap dump"):
        sys.exit(f"not a go1.7 heap dump: {hdr!r}")
    r = R(fh)
    counts = {}
    while True:
        try:
            kind = r.uvarint()
        except EOFError:
            break
        counts[kind] = counts.get(kind, 0) + 1
        if kind == 0:
            break
        elif kind == 1:      # object: addr, contents, fields
            addr = r.uvarint()
            size = r.skip_string()
            r.fieldlist()
            on_object(addr, size)
        elif kind == 2:      # otherroot: description, pointer
            r.skip_string(); r.uvarint()
        elif kind == 3:      # type: addr, size, name, indirect
            r.uvarint(); r.uvarint(); r.skip_string(); r.uvarint()
        elif kind == 4:      # goroutine: 8 ints, waitreason string, 4 ints (runtime/heapdump.go)
            for _ in range(8):
                r.uvarint()
            r.skip_string()
            for _ in range(4):
                r.uvarint()
        elif kind == 5:      # stack frame
            r.uvarint(); r.uvarint(); r.uvarint()
            r.skip_string()
            r.uvarint(); r.uvarint(); r.uvarint()
            r.skip_string()
            r.fieldlist()
        elif kind == 6:      # dump params
            r.uvarint()      # bigendian
            for _ in range(3):
                r.uvarint()
            r.skip_string()  # arch
            r.skip_string()  # GOEXPERIMENT
            r.uvarint()      # ncpu
        elif kind == 7:      # registered finalizer
            for _ in range(5):
                r.uvarint()
        elif kind == 8:      # itab
            r.uvarint(); r.uvarint()
        elif kind == 9:      # os thread
            r.uvarint(); r.uvarint(); r.uvarint()
        elif kind == 10:     # mem stats: 24 ints + PauseNs[256] + NumGC, fixed 281 ints
            for _ in range(281):
                r.uvarint()
        elif kind == 11:     # queued finalizer: same five fields as a registered one
            for _ in range(5):
                r.uvarint()
        elif kind in (12, 13):  # data / bss segment
            r.uvarint()
            r.skip_string()
            r.fieldlist()
        elif kind == 14:     # defer record: d, gp, sp, pc, fn, fn.fn, link
            for _ in range(7):
                r.uvarint()
        elif kind == 15:     # panic record: p, gp, type, data, 0, link
            for _ in range(6):
                r.uvarint()
        elif kind == 16:     # alloc/free profile
            pid = r.uvarint()
            psize = r.uvarint()   # bucket's allocation size; lets object records be skipped (see main)
            nf = r.uvarint()
            frames = []
            for _ in range(nf):
                fn = r.string().decode(errors="replace")
                r.skip_string()  # file
                r.uvarint()      # line
                frames.append(fn)
            r.uvarint(); r.uvarint()  # allocs, frees
            on_profile(pid, frames, psize)
        elif kind == 17:     # alloc sample: addr, profile id
            # Read in two statements: a one-liner evaluates the right side first, reading the
            # bucket before the address and storing the pair inverted (once caused 0 of 86 joins).
            a = r.uvarint()
            on_sample(a, r.uvarint())
        else:
            sys.exit(f"unknown record kind {kind} at offset {r.tell()}; refusing to guess: "
                     f"a mis-parsed record would silently corrupt every following one")

    fh.close()
    return counts


def site(frames):
    # The leaf frames are allocator internals; the first frame outside runtime/mallocgc is
    # the allocating function, the campaign's grouping key.
    for f in frames:
        if not (f.startswith("runtime.") or f.startswith("internal/")):
            return f
    return frames[0] if frames else "?"


def main():
    dump, out = sys.argv[1], sys.argv[2]
    with_unsampled = "--with-unsampled" in sys.argv[3:]

    # One pass by default: profile records alone give every sampled object's address, size and
    # stack. A second pass (--with-unsampled, off by default, ~3 more hours on a 570 GB dump)
    # additionally locates unsampled objects so join_residency.py can charge their pages to
    # "unsampled" instead of leaving them unaccounted.
    samples = {}
    profiles = {}
    counts = walk(dump,
                  on_object=lambda a, sz: None,
                  on_sample=lambda a, pid: samples.__setitem__(a, pid),
                  on_profile=lambda pid, fr, sz: profiles.__setitem__(pid, (fr, sz)))
    print(f"records by kind: {dict(sorted(counts.items()))}")
    print(f"{len(samples)} sampled objects, {len(profiles)} distinct allocation buckets")

    n = 0
    total = 0
    with open(out, "w") as o:
        for addr, pid in samples.items():
            got = profiles.get(pid)
            if got is None:
                continue
            fr, sz = got
            n += 1
            total += sz
            o.write(json.dumps({"addr": addr, "size": sz,
                                "site": site(fr), "stack": fr[:STACK_KEEP]}) + "\n")
        print(f"wrote {n} sampled objects, {total} bytes")

        if with_unsampled:
            stat = {"n": 0, "bytes": 0}

            def emit(addr, size):
                if addr in samples:
                    return
                stat["n"] += 1
                stat["bytes"] += size
                o.write(json.dumps({"addr": addr, "size": size, "site": "unsampled"}) + "\n")

            walk(dump, on_object=emit,
                 on_sample=lambda a, p: None, on_profile=lambda p, f, s: None)
            print(f"second pass: {stat['n']} unsampled objects, {stat['bytes']} bytes "
                  f"({100.0 * total / (total + stat['bytes']):.1f}% of object bytes carry a site)"
                  if (total + stat["bytes"]) else "second pass: no objects")
        else:
            print("unsampled objects NOT enumerated (pass --with-unsampled for a second pass); "
                  "join_residency.py will report their pages as unaccounted rather than as "
                  "'unsampled'")


if __name__ == "__main__":
    main()
