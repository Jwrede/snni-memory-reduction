#!/usr/bin/env python3
"""PUMA s2: the distributed object store never collects; its GC threshold of 50 is unreachable for this workload, so lower it to 1.

    patch_gc_threshold.py <path to distributed_impl.py>
"""
import os
import re
import sys

ANCHOR = "    _GC_COLLECT_THRESHOLD = 50\n"

PATCH = '''    # s2_gc_threshold: 50 was never reachable here. PUMA's run creates a handful of object
    # refs -- the parameter set, three encoded inputs, the result -- so `len(self._dead_refs)`
    # never approaches fifty, `_garbage_collect` returns immediately every time, and the model
    # stays resident on P2 for the whole run. Measured: 437,935,128 B of jax arrays on the
    # binding party, 6.7% of the peak, against a recorder row of 437,739,520 B.
    #
    # The release machinery is the program's own and is correct. Only the constant is wrong for
    # this workload.
    _GC_COLLECT_THRESHOLD = 1
'''


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    if not os.path.exists(src):
        sys.exit("FAILED: no file at " + src)
    txt = open(src).read()

    if "s2_gc_threshold" in txt:
        sys.exit("FAILED: this file already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the threshold matched {txt.count(ANCHOR)} times, expected exactly 1")

    # Verify the release chain _dec_objref -> _garbage_collect -> builtin_gc -> _del_object exists;
    # the threshold change is worthless if the chain is absent.
    for name in ("_garbage_collect", "builtin_gc", "_del_object"):
        if name not in txt:
            sys.exit(f"FAILED: `{name}` is missing; the release chain this step unblocks is not "
                     "the one in this file")
    # `_dec_objref` is defined twice; anchor from the THRESHOLD's position so we land on
    # `HostContext._dec_objref` (the one that calls `_garbage_collect`), not the delegating stub.
    at = txt.index(ANCHOR)
    dec_at = txt.find("def _dec_objref", at)
    if dec_at < 0:
        sys.exit("FAILED: no `_dec_objref` after the threshold; the threshold is not in the class "
                 "that does the bookkeeping")
    dec = txt[dec_at:]
    nxt = re.search(r"\n    def ", dec)
    if nxt:
        dec = dec[:nxt.start()]
    if "_garbage_collect()" not in dec:
        sys.exit("FAILED: the `_dec_objref` that belongs to this threshold does not call "
                 "`_garbage_collect`; lowering it would change nothing")

    txt = txt.replace(ANCHOR, PATCH, 1)
    txt = txt + '''

def _snni_gc_threshold_marker():
    import sys as _sys
    print("LEVER|gc_threshold|armed value=1", file=_sys.stderr, flush=True)


_snni_gc_threshold_marker()
'''
    open(src, "w").write(txt)
    print("patch_gc_threshold: the object store now collects at one dead ref instead of fifty")


if __name__ == "__main__":
    main()
