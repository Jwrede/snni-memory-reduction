#!/usr/bin/env python3
"""PUMA s1: pickle the response directly into the 10 MiB send pieces (bounded queue, one worker
thread) instead of building it whole. Free: intermediate that never had to exist whole.

Usage: python3 patch_response_stream.py <path to distributed_impl.py>
"""
import os
import re
import sys

ANCHOR = """        for split in split_message(response):
            yield distributed_pb2.RunResponse(data=split)
"""

HELPER = '''

class _SnniChunkSink:
    """s1_response_stream: a file-like that emits finished CHUNK_SIZE pieces as they are pickled.

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

    q = _queue.Queue(maxsize=4)
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
def _snni_response_stream_marker():
    import sys as _sys
    print("LEVER|response_stream|armed", file=_sys.stderr, flush=True)


_snni_response_stream_marker()
'''

PATCH = """        # s1_response_stream: cut the pickle into pieces AS IT IS WRITTEN. The stock code built
        # `response` whole, which is the `_PyBytes_Resize` row of this system's peak table, and then
        # sliced it into exactly these pieces anyway. Same bytes, same order, so the receiver is
        # unchanged; what disappears is the complete second copy of the result.
        for split in _snni_stream_pickle(result):
            yield distributed_pb2.RunResponse(data=split)
"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    if not os.path.exists(src):
        sys.exit("FAILED: no file at " + src)
    txt = open(src).read()

    if "s1_response_stream" in txt:
        sys.exit("FAILED: this file already carries the lever")

    # The send loop appears twice (RunReturn and Run); only RunReturn may be streamed (Run's
    # response is 2 kB of object refs). Require exactly two matches.
    n = txt.count(ANCHOR)
    if n != 2:
        sys.exit(f"FAILED: the send loop matched {n} times, expected exactly 2 (RunReturn and Run)")

    # RunReturn's loop is preceded by `cloudpickle.dumps(result)`; anchor on that, not position.
    first = txt.index(ANCHOR)
    before = txt[:first]
    if "cloudpickle.dumps(result)" not in before:
        sys.exit("FAILED: the first send loop is not the one that pickles `result`; this patch was "
                 "written against RunReturn coming first")
    if before.rindex("cloudpickle.dumps(result)") < before.rindex("def RunReturn"):
        sys.exit("FAILED: `cloudpickle.dumps(result)` is not inside RunReturn")

    # RunReturn's exception branch also assigns `response` and needs the stock loop; verify shape.
    if "response = cloudpickle.dumps(Exception(stack_info))" not in before:
        sys.exit("FAILED: RunReturn's exception branch does not have the shape this patch expects")

    txt = txt[:first] + PATCH + txt[first + len(ANCHOR):]

    # Re-insert the stock send loop at the end of the except block for the error path.
    exc_anchor = "            response = cloudpickle.dumps(Exception(stack_info))\n"
    at = txt.index(exc_anchor)
    if txt.count(exc_anchor) != 2:
        sys.exit(f"FAILED: the exception branch matched {txt.count(exc_anchor)} times, expected 2")
    txt = (txt[:at + len(exc_anchor)]
           + "            # s1_response_stream: the error path keeps the stock send, because an\n"
             "            # exception's pickle is a few kilobytes and a thread for it would be\n"
             "            # pure overhead.\n"
             "            for split in split_message(response):\n"
             "                yield distributed_pb2.RunResponse(data=split)\n"
             "            return\n"
           + txt[at + len(exc_anchor):])

    txt = txt + HELPER
    open(src, "w").write(txt)
    print("patch_response_stream: RunReturn streams its pickle in CHUNK_SIZE pieces")


if __name__ == "__main__":
    main()
