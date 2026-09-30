# p_runtimeconfig: SPU RuntimeConfig probes

Probes `p_chunk16` (46125204), `p_conc1` (46125635), `p_nommulsplit` (46125666), one run each, stock
image and policy, config via `PUMA_CONFIG` (bind-mounted, no image build).

```
share_max_chunk_size                       = 134217728 (128 MiB)   probed
max_concurrency                            = 0 (unlimited)         probed
experimental_disable_mmul_split            = False                 probed
experimental_enable_colocated_optimization = False                 NOT probed, see below
```

`experimental_enable_colocated_optimization` is off-path: the measured cluster is five parties plus
a driver on `127.0.0.1`, a stand-in for a distributed deployment.

| | peak kB | wall s | vs s0 peak |
|---|---:|---:|---|
| `s0_default` | 7,320,212 | 357 | baseline, spread 0.882% over three runs |
| `p_chunk16` (chunk 128 -> 16 MiB) | 7,403,312 | 366 | +1.1%, worse |
| `p_conc1` (`max_concurrency` 0 -> 1) | 7,239,264 | 555 | -1.11%, wall +55% |
| `p_nommulsplit` (mmul split off) | 7,312,104 | 359 | inside the baseline range |

- Gates within the baseline's control range.
- `p_conc1`: memory per run time 0.02 (BOLT `s2_threads_4`: 0.055); one replicate.

## Invariant objects across seven programs

```
VARIANT             peak kB     pybind11::bytes    _PyBytes_Resize
s0_default          7,320,212     5,250,162,688      1,125,175,296
p_chunk16           7,403,312     5,250,998,272      1,125,175,296
p_conc1             7,239,264     5,250,158,592      1,125,179,392
p_nommulsplit       7,312,104     5,250,158,592      1,125,175,296
p_noreload          7,287,564     5,250,158,592      1,125,175,296
s1_response_stream  7,523,380     5,250,158,592      1,125,175,296
s1_request_stream   7,341,252     5,250,158,592      1,125,175,296
```

Peak varies 284 MB; `pybind11::bytes` by 839,680 B (0.016%); `_PyBytes_Resize` by 4,096 B.

## `p_chunk16` on `s1_response_stream_v2`

Fast census at 99.0% of the v2 peak:

```
134,217,744 B  list   x6      = 128 MiB + 16, i.e. exactly `share_max_chunk_size`
106,618,897 B  stack distributed_impl.py:write:1426 b   x3   (v2's own sink)
 10,485,760 B  split / queue item / deque / dict        several
```

```
v2 (128 MiB)   6,381,368 / 6,410,776 / 6,413,628 kB   median 6,410,776
p_chunk16_on_v2 (16 MiB)   6,541,284 kB               +2.0%, worse
pybind11::bytes  5,250,158,592 -> 5,250,994,176       +0.016%, unchanged
```

## The sink's 320 MB is borrowed

`b` in `_SnniChunkSink.write` is a `bytes`. CPython's pickler frames output at 64 KiB; a 106 MB
`write` comes from the large-data bypass, which passes the original object:

```
pickle.Pickler(sink, protocol=5).dump({"w": <100 MiB bytes>, "n": 7})
  original id: 139449618788368  len 104857600
  write() bekam id=139449618788368 type=bytes len=104857600  IDENTISCH=True
```

The 320 MB are the payload in transit, also present in the stock program.

## Accounting at the v2 peak

| item | size | status |
|---|---:|---|
| libspu `pybind11::bytes` | 5.25 GB | wheel |
| plaintext model on P2 | 438 MB | live by construction (`s2_gc_threshold/`) |
| payload in transit | 320 MB | borrowed by the sink |
