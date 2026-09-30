#!/usr/bin/env python3
"""PUMA s1 (v2): RunReturn materialises the whole response AND streams it; pickle straight into the pieces so the whole never exists. v1 removed only the send loop, leaving the double materialisation.

Usage: python3 patch_response_stream.py <path to distributed_impl.py>
"""
import os
import re
import sys

ANCHOR = """        try:
            from jax.tree_util import tree_map

            args, kwargs = tree_map(lambda obj: self._get_object(obj), (args, kwargs))
            result = fn(self, *args, **kwargs)
            response = cloudpickle.dumps(result)
        except Exception as e:
            stack_info = traceback.format_exc()
            logger.info(stack_info)
            response = cloudpickle.dumps(Exception(stack_info))
        for split in split_message(response):
            yield distributed_pb2.RunResponse(data=split)
"""

PATCH = """        try:
            from jax.tree_util import tree_map

            args, kwargs = tree_map(lambda obj: self._get_object(obj), (args, kwargs))
            result = fn(self, *args, **kwargs)
            # s1_response_stream_v2: NO `response = cloudpickle.dumps(result)` here, and that line
            # is the whole difference to v1. v1 replaced only the SEND LOOP and left the
            # materialisation, so the program pickled `result` twice: once into a `response` nobody
            # read and once through the stream. The fast census caught it in the very image that
            # claimed to have removed it: three 375,054,672 B copies of `response` still in this
            # frame at 98.3% of the peak, with the stream's own buffers on top.
        except Exception as e:
            stack_info = traceback.format_exc()
            logger.info(stack_info)
            response = cloudpickle.dumps(Exception(stack_info))
            # The error path keeps the stock send: an exception's pickle is a few kilobytes and a
            # worker thread for it would be pure overhead. It returns here so the success path
            # below never sees `response`.
            for split in split_message(response):
                yield distributed_pb2.RunResponse(data=split)
            return
        # Pickle STRAIGHT INTO the pieces gRPC sends. The whole never exists.
        for split in _snni_stream_pickle(result):
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



def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    if not os.path.exists(src):
        sys.exit("FAILED: no file at " + src)
    txt = open(src).read()

    if "s1_response_stream" in txt:
        sys.exit("FAILED: this file already carries the lever")
    # Replace materialisation, exception branch and send loop as ONE block; v1 anchored the send
    # loop alone and left the materialisation behind.
    if txt.count(ANCHOR) != 1:
        sys.exit(f"FAILED: RunReturn's try/except/send block matched {txt.count(ANCHOR)} times, "
                 "expected exactly 1")

    txt = txt.replace(ANCHOR, PATCH, 1)

    # Guard v1 lacked: on the success path `response` must no longer exist (v1 checked `payload`).
    fn_body = txt[txt.index("def RunReturn"):]
    nxt = re.search(r"\n    def ", fn_body)
    if nxt:
        fn_body = fn_body[:nxt.start()]
    after_return = fn_body[fn_body.rindex("            return") + len("            return"):]
    after_return = re.sub(r"#[^\n]*", "", after_return)
    if re.search(r"\bresponse\b", after_return):
        sys.exit("FAILED: `response` survives on the success path; that is the v1 defect")
    if "_snni_stream_pickle(result)" not in fn_body:
        sys.exit("FAILED: the stream call is not in RunReturn")

    txt = txt + HELPER
    open(src, "w").write(txt)
    print("patch_response_stream_v2: RunReturn no longer materialises `response` at all")


if __name__ == "__main__":
    main()
