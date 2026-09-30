# s1_response_stream: retracted, never removed `response`

Jobs 46120622-24, 3 replicates, image layered on the baseline. Patch beside this file.

```
s0      7,320,212 / 7,337,192 / 7,272,656 kB   median 7,320,212   spread 0.882%
stream  7,523,380 / 7,555,696 / ...            about +3%, outside the spread
peak instant   s0 24.0% and 23.8% of the run   stream 24.5% and 25.2%   unchanged
```

Gate: s0 against itself 2.560e-03 to 5.180e-03; step against s0 1.953e-03 to 3.113e-03.

## Defect (found 2026-08-14)

The patch replaced the send loop and kept the materialisation:

```python
result = fn(self, *args, **kwargs)
response = cloudpickle.dumps(result)      # untouched
...
for split in _snni_stream_pickle(result): # and then pickles it a SECOND time
```

Fast census in this image, 98.3% of the peak:

```
375,054,672 B  frame distributed_impl.py:RunReturn:298 response
375,054,672 B  frame distributed_impl.py:RunReturn:298 response
375,054,672 B  frame distributed_impl.py:RunReturn:298 response
134,217,744 B  stack distributed_impl.py:write:1424 b        <- the stream's own sink
```

- The guard checked `payload`, not that `response` became dead.
- Withdrawn: "it streamed the wrong one of two `dumps` sites"; `s1_request_stream`, derived from that
  reading, is void.
- Retried as `off-path-runs/s1_response_stream_v2` (replaces the whole try/except/send block; refuses
  to write unless `response` is gone from the success path).

## Object table (same runs)

```
                    s0_default                 s1_response_stream
4.89 GiB  pybind11::bytes::bytes             4.89 GiB   unchanged
1.05 GiB  ... <- _PyBytes_Resize             1.05 GiB   UNCHANGED
0.41 GiB  ... <- PyBytes_FromStringAndSize   0.41 GiB   unchanged
                                             0.18 GiB   NEW: the queued chunks
```

The new 0.18 GiB (bounded queue and chunks in flight, 193 MB) accounts for the +203 MB on the peak.
The streaming helper itself works: byte-identical output, same chunk count, no deadlock on early
close.
