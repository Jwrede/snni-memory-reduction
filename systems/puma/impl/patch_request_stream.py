#!/usr/bin/env python3
"""PUMA s1: the request is pickled whole before it is sent in 10 MiB pieces; stream the pickle straight into the pieces instead.

Usage: python3 patch_request_stream.py <path to distributed_impl.py>
"""
import os
import re
import sys

ANCHOR = """        payload = cloudpickle.dumps((fn, args, kwargs))
        rsp_gen = stub_method(
            distributed_pb2.RunRequest(data=split) for split in split_message(payload)
        )
"""

PATCH = """        # s1_request_stream: pickle STRAIGHT INTO the pieces gRPC is about to send. The stock
        # code built `payload` whole, which is the `_PyBytes_Resize` row of this system's peak
        # table, and then cut it into exactly these pieces anyway. The consumer here is already a
        # generator, so the whole never has to exist.
        rsp_gen = stub_method(
            distributed_pb2.RunRequest(data=split)
            for split in _snni_stream_pickle((fn, args, kwargs))
        )
"""

HELPER = '''

class _SnniChunkSink:
    """s1_request_stream: a file-like that emits finished CHUNK_SIZE pieces as they are pickled.

    The stock code built the whole serialization and then sliced it. This cuts while it is written,
    so the whole never exists. The queue is bounded, which is the entire point: without a bound the
    chunks would simply accumulate and the peak would not move.
    """

    def __init__(self, q):
        self._q = q
        self._buf = bytearray()

    def write(self, b):
        # NEVER COPY A LARGE WRITE INTO THE CARRY BUFFER, and this is the whole difference between
        # a step that works and one that made the peak 8.5% WORSE. `pickle` hands a big `bytes`
        # straight through: measured in this image, a 250 MB object arrives as ONE write call of
        # 262,144,000 bytes. The first version did `self._buf += b`, so the pickler's argument and
        # the carry buffer held the same 250 MB at the same time. Slicing out of `b` through a
        # memoryview instead means the only copy made here is the chunk that is handed on, which
        # the consumer needs anyway.
        n = RPC.CHUNK_SIZE
        mv = memoryview(b)
        if self._buf:
            take = min(n - len(self._buf), len(mv))
            self._buf += mv[:take]
            mv = mv[take:]
            if len(self._buf) == n:
                self._q.put(bytes(self._buf))
                self._buf = bytearray()
        while len(mv) >= n:
            self._q.put(bytes(mv[:n]))
            mv = mv[n:]
        if mv:
            self._buf += mv
        return len(b)

    def flush(self):
        if self._buf:
            self._q.put(bytes(self._buf))
            self._buf = bytearray()


def _snni_stream_pickle(obj):
    """Yield the pickle of `obj` in CHUNK_SIZE pieces, in order, without ever holding it whole.

    `dump()` runs to completion before returning, so a generator cannot yield from inside it: the
    pickler runs in one worker thread and this generator drains its queue. The byte stream is the
    concatenation of the same chunks in the same order as `split_message(cloudpickle.dumps(obj))`,
    so the receiving side is unchanged.
    """
    import queue as _queue
    import threading as _threading

    q = _queue.Queue(maxsize=2)
    done = object()

    def _work():
        sink = _SnniChunkSink(q)
        try:
            cloudpickle.CloudPickler(sink).dump(obj)
            sink.flush()
        except BaseException as exc:      # carried across the thread boundary, never swallowed
            q.put(exc)
        else:
            q.put(done)

    t = _threading.Thread(target=_work, name="snni-pickle", daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # IF THE CONSUMER STOPS EARLY the worker may be blocked on a full queue, and a plain join()
        # would then wait forever. gRPC closes a response generator on a cancelled RPC, so this is
        # a real path and not a hypothetical: drain until the worker exits.
        while t.is_alive():
            try:
                q.get_nowait()
            except _queue.Empty:
                t.join(timeout=0.05)


# Announced at import, so a run can prove the lever is in the image it measured. `sys` is imported
# locally: this module does not import it at the top and the patch does not add imports it does not
# need, because every added name is one more thing that can collide.
def _snni_request_stream_marker():
    import sys as _sys
    print("LEVER|request_stream|armed", file=_sys.stderr, flush=True)


_snni_request_stream_marker()
'''



def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    if not os.path.exists(src):
        sys.exit("FAILED: no file at " + src)
    txt = open(src).read()

    if "s1_request_stream" in txt:
        sys.exit("FAILED: this file already carries the lever")
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: the client send matched {txt.count(ANCHOR)} times, expected exactly 1")

    # Refuse if `payload` is read elsewhere in `_call`: it ceases to exist, so a reader would raise.
    at = txt.index(ANCHOR)
    body = txt[at + len(ANCHOR):]
    nxt = re.search(r"\n    def ", body)
    if nxt:
        body = body[:nxt.start()]
    body = re.sub(r"#[^\n]*", "", body)
    if re.search(r"\bpayload\b", body):
        sys.exit("FAILED: `payload` is read again later in _call; removing it would break a reader")

    txt = txt[:at] + PATCH + txt[at + len(ANCHOR):]
    txt = txt + HELPER
    open(src, "w").write(txt)
    print("patch_request_stream: the client pickles straight into the pieces it sends")


if __name__ == "__main__":
    main()
