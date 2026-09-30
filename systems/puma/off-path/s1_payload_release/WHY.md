# s1_payload_release: null result

Jobs 46120521-23, 3 replicates, image `puma-s1_payload_release.sif` (layer on the baseline image).
Patch beside this file. `LEVER|payload_release|armed` in all runs; the build required `del payload`
in both server methods.

```
s0   7,320,212 / 7,337,192 / 7,272,656 kB   median 7,320,212   spread 0.882%
s1   7,295,616 / 7,305,540 / 7,337,168 kB   median 7,305,540   spread 0.569%

median difference   -14,672 kB = -0.200%     the ranges overlap completely
predicted           -385 MB    = -5.14%
```

Gate (worst absolute difference):

```
s0r1 vs s0r2  5.180e-03      s1r1 vs s0r1  3.109e-03
s0r1 vs s0r3  4.307e-03      s1r1 vs s0r2  3.986e-03
s0r2 vs s0r3  2.560e-03      s1r1 vs s0r3  3.113e-03
```

## Peak instant (`vmrss.log`)

```
s0 run1   peak at 24.0% of the run
s1 run1   peak at 25.3%
s1 run2   peak at 24.2%
```

The peak is in the infeed (first quarter of the run).

## Census at 93.5% of the binding party's peak

```
750,109,344 B  n=2  bytes  held_by=frame distributed_impl.py:RunReturn:298 response
 20,971,530 B  n=2  bytes  held_by=frame _server.py:_send_response:708 serialized_response
 20,971,520 B  n=2  bytes  held_by=frame distributed_impl.py:RunReturn:298 split
      2,154 B  n=3  bytes  held_by=frame distributed_impl.py:RunReturn:298 payload
```

- `payload` is 2,154 B: the request carries SPU object references (uuid, node id), not data.
- The ~385 MB is `response`, the pickled result (n=2: two `RunReturn` calls in flight); the recorder's
  rank two `_PyBytes_Resize` is pickle growing that buffer.
- Next candidate at the time: stream the pickle into the 10 MiB chunks `split_message` produces
  (`s1_response_stream`).
