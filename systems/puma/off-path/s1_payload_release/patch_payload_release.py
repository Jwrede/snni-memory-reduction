#!/usr/bin/env python3
"""PUMA s1: `del payload` after its last use in Run and RunReturn (the serialized request was held
across the whole SPU computation). Free: dead object retained past last use.

Usage: python3 patch_payload_release.py <path to distributed_impl.py>
"""
import os
import re
import sys

ANCHOR = """        payload = rebuild_messages(itr.data for itr in req_itr)
        # Warning: this is only a demo, do not use in production.
        (fn, args, kwargs) = pickle.loads(payload)
"""

PATCH = """        payload = rebuild_messages(itr.data for itr in req_itr)
        # Warning: this is only a demo, do not use in production.
        (fn, args, kwargs) = pickle.loads(payload)
        # s1_payload_release: the request is dead here. Its last reader is the unpickle above, and
        # CPython keeps a local bound until the function returns, so the stock code holds the whole
        # serialized request resident across fn(...), i.e. across the entire SPU computation, which
        # is where this system's peak falls.
        del payload
"""

LEVER = '''

def _snni_payload_release_marker():
    """s1_payload_release: announced once per process so the run can prove the lever is in."""
    import sys as _sys
    print("LEVER|payload_release|armed", file=_sys.stderr, flush=True)


_snni_payload_release_marker()
'''


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    if not os.path.exists(src):
        sys.exit("FAILED: no file at " + src)
    txt = open(src).read()

    if "s1_payload_release" in txt:
        sys.exit("FAILED: this file already carries the lever")
    # Both server methods (RunReturn and Run) bind the full request; require exactly two matches.
    n = txt.count(ANCHOR)
    if n != 2:
        sys.exit(f"FAILED: the request-unpickle shape matched {n} times, expected exactly 2 "
                 "(RunReturn and Run)")

    # Refuse if `payload` is read after the unpickle in either method (checked against the file).
    for m in re.finditer(re.escape(ANCHOR), txt):
        body = txt[m.end():]
        nxt = re.search(r"\n    def ", body)
        if nxt:
            body = body[:nxt.start()]
        body = re.sub(r"#[^\n]*", "", body)
        uses = re.findall(r"\bpayload\b", body)
        if uses:
            sys.exit(f"FAILED: `payload` is read {len(uses)} more time(s) after its unpickle; "
                     "deleting it there would break a live reader")

    txt = txt.replace(ANCHOR, PATCH)
    txt = txt + LEVER
    open(src, "w").write(txt)
    print("patch_payload_release: the serialized request is released at its last use in Run and "
          "RunReturn")


if __name__ == "__main__":
    main()
