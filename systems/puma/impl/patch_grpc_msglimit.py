#!/usr/bin/env python3
"""PUMA: the gRPC message limits are sized to the largest message instead of to 1 GiB. BORROWED.

    patch_grpc_msglimit.py <distributed_impl.py> <bytes>
"""
import sys

ANCHOR = """    OPTIONS = [
        ("grpc.max_message_length", 1024 * 1024 * 1024),
        ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
        ("grpc.so_reuseport", 0),
    ]"""


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    path, n = sys.argv[1], int(sys.argv[2])
    txt = open(path).read()
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the OPTIONS block matched {txt.count(ANCHOR)} times, expected 1")
    if "CHUNK_SIZE = 10 * 1024 * 1024" not in txt:
        sys.exit("FAILED: CHUNK_SIZE is not 10 MiB in this file; the limit below was derived from "
                 "it and must not be transplanted onto a different chunking")
    if n < 10 * 1024 * 1024:
        sys.exit(f"FAILED: {n} is below CHUNK_SIZE; a message would be refused rather than sized")
    new = f"""    # BORROWED from SIGMA-GPU's measured-buffer family. Every message here is a chunk and
    # CHUNK_SIZE is 10 MiB, so 1 GiB was a hundredfold margin on a constant of the program.
    OPTIONS = [
        ("grpc.max_message_length", {n}),
        ("grpc.max_receive_message_length", {n}),
        ("grpc.so_reuseport", 0),
    ]"""
    txt = txt.replace(ANCHOR, new, 1)
    if "LEVER|grpc_msglimit" not in txt:
        txt += ('\n\nimport sys as _snni_sys2\n'
                f'print("LEVER|grpc_msglimit|bytes={n}", file=_snni_sys2.stderr, flush=True)\n')
    open(path, "w").write(txt)
    print(f"patch_grpc_msglimit: gRPC message limits 1 GiB -> {n} B "
          f"({n / (1024 * 1024):.0f} MiB, {n / (10 * 1024 * 1024):.1f}x CHUNK_SIZE)")


if __name__ == "__main__":
    main()
