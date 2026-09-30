#!/usr/bin/env python3
"""PUMA: fewer concurrent RPC handlers, so fewer responses coexist. BORROWED, PAID.

    patch_rpc_workers.py <distributed_impl.py> <n>
"""
import re
import sys

ANCHOR = "concurrent.futures.ThreadPoolExecutor(max_workers=10)"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    path, n = sys.argv[1], int(sys.argv[2])
    txt = open(path).read()
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the server's thread pool matched {txt.count(ANCHOR)} times, expected 1")
    new = f"concurrent.futures.ThreadPoolExecutor(max_workers={n})"
    txt = txt.replace(ANCHOR, new, 1)
    # Marker announced from the patched file, not this script, so it reports the patched code ran.
    marker = ('\n\nimport sys as _snni_sys\n'
              f'print("LEVER|rpc_workers|max_workers={n}", file=_snni_sys.stderr, flush=True)\n')
    if "LEVER|rpc_workers" not in txt:
        txt += marker
    open(path, "w").write(txt)
    print(f"patch_rpc_workers: server thread pool 10 -> {n}")


if __name__ == "__main__":
    main()
