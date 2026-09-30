# p_census: identifying the 5.36 GB allocation

Diagnostic runs; none is evidence for a step. Census armed in every Python process via
`SNNI_PUMA_CENSUS=1` (`impl/snni_census_sitecustomize.py`).

## Run 1 (2026-08-04, job 45757960): arrays only

Binding party pid 432947; last census block at RSS 7,271,100 kB (0.1% below the published peak).

```
115.6 MB  49x jax float32 (768,768)      the attention projections
113.2 MB  12x jax float32 (3072,768)     FFN down
113.2 MB  12x jax float32 (768,3072)     FFN up
 93.8 MB   1x jax float32 (30522,768)    the embedding table
  1.6 MB   1x jax float32 (512,768)      positions
  0.5 MB   the biases and scalars
```

- 437.9 MB = one float32 copy of the 109,480,704-parameter model, constant while RSS climbs 6.03 ->
  7.27 GB.
- The 5,761,327,104 B allocation (`heap:python3.10+0x13cc27`, 77%) is no live numpy/jax array.
- Arithmetic: share transfer `109,480,704 params x 8 B x 3 parties x 2 (serialize + copy) = 5.26 GB`,
  within 2%. Size 720,165,888 x 8 B (= 2^10 x 3^2 x 13 x 6011): a growth buffer, not one tensor.
- Working identification: serialization arena of the parameter-share transfer into the SPU runtime,
  on P2.

Depth test (diagnostic driver with `num_hidden_layers` passed to both `from_pretrained` calls; the
measured driver drops trailing layers from `params` but not from the config, so flax raises
`ScopeParamNotFoundError` below 12):

| | peak (kB) |
|---|---:|
| 12 layers (`../p_depth12`) | 7,265,316 (0.2% from the baseline) |
| 6 layers (`../p_depth6`) | 5,076,812 |
| difference | 2,188,504, against 2.04 GB predicted (6 x 7.08M params x 8 B x 3 x 2) |

## Run 2 (2026-08-14, job 46120493): bytes visible

- `gc.get_objects()` cannot return `bytes` (not gc-tracked). The walk now follows `gc.get_referents`
  from gc-tracked objects to depth 8 and records each `bytes`/`bytearray` with its holder type
  (verified on a fixture: dict, list, bytearray).
- `express`, policy `2026-07-29.1` (`pm_min=65536 fullscan=1 freeze=1 poll_ms=15`), wall 223 s,
  `polled_peak_kb=7,344,524`.
- Symboliser now names the row `pybind11::bytes::bytes<unsigned long,0>(char const*, unsigned long const&) <- PyBytes_FromStringAndSize` (creator, not holder).

Binding party pid 548481, highest block `rss_kb=6,487,076`:

```
437,935,128 B   jax arrays, eleven groups     unchanged from 04.08 to the byte
385,541,868 B   n=4    bytes  held_by=frame
      8,802 B   n=373  bytes  held_by=dict
      1,798 B   n=151  bytes  held_by=list
        680 B   n=20   bytes  held_by=_StreamStreamMultiCallable
        158 B   n=7    bytes  held_by=Pattern
         74 B   n=26   bytes  held_by=Struct
         46 B   n=10   bytes  held_by=tuple
```

`gc.get_objects()` returns no frames in CPython 3.10.20, so running stacks were outside the walk.

| repair | wall s | peak kB | verdict |
|---|---:|---:|---|
| baseline, no census (`s0_default`) | 357 | 7,320,212 | |
| census, gc-rooted walk only | 223 | 7,344,524 | +0.33% |
| census, frames seeded into the frontier | 477 | 11,916,752 | withdrawn: pins the jax/grpc graphs |
| separate shallow frame scan (own referents plus one level; `f_globals`, `f_builtins`, own thread skipped) | | | kept; fixture: 120 MiB frame local and 40 MiB dict local attributed correctly, 10 MiB module buffer not; +1.4% wall |

## Depth probes (recorder)

| probe | `pm_depth` | rank-one caller | wall s | peak kB |
|---|---:|---|---:|---:|
| `s0_default` (baseline) | 2 | `pybind11::bytes::bytes<unsigned long,0>(char const*, unsigned long const&)` | 357 | 7,320,212 |
| `p_pmdepth5` | 5 | `allocator_traits<allocator<pybind11::bytes>>::construct<pybind11::bytes, std::string>` | 449 | 7,340,328 |
| `p_pmdepth8` | 8 | `std::vector<pybind11::bytes>::emplace_back<std::string>(std::string&&)` | 483 | 7,331,984 |
| `p_pmdepth11` | 11 | `std::vector<pybind11::bytes>::emplace_back<std::string>(std::string&&)` | 479 | 7,299,724 |

```
4.89 GiB  75.6%  std::allocator_traits<std::allocator<pybind11::bytes> >::construct<
                     pybind11::bytes, std::string>(
                     std::allocator<pybind11::bytes>&, pybind11::bytes*, std::string&&)
                 <- PyBytes_FromStringAndSize
1.05 GiB  16.2%  _PyFunction_Vectorcall  <- _PyBytes_Resize
0.41 GiB   6.3%  python3.10+0x15031b     <- PyBytes_FromStringAndSize
```

- The chain bottoms out at depth 8 inside libspu: a `std::vector<pybind11::bytes>` filled one element
  per `std::string`. The 4.89 GiB are CPython `bytes` whose only reference at the peak is that C++
  vector.
- Depth does not change the peak (within 0.55%); it costs wall time (+26% at depth 5, +35% at 8 and 11).
- `impl/puma.def` installs `spu==0.9.5` as a wheel; a C++ fix means a source build of SPU and a new
  baseline.

## Run 3 (2026-08-14, job 46120509): two-walk census

Census cost: peak 7,320,212 -> 7,467,704 kB (+2.0%, above the 0.226% spread); sixteen blocks on the
binding party.

Highest block, RSS 6,628,676 kB (88.8%):

```
437,935,128 B  jax arrays, eleven groups        unchanged across all three census runs
385,566,719 B  n=5  bytes  held_by=frame distributed_impl.py:RunReturn:298
 31,457,295 B  n=3  bytes  held_by=frame _server.py:_send_response:708
      8,803 B  n=373 bytes held_by=dict
```

`/opt/puma/examples/python/utils/distributed_impl.py` (SPU examples at `02ac612d`, cloned separately,
not in the wheel):

```python
def RunReturn(self, req_itr, ctx):
    payload = rebuild_messages(itr.data for itr in req_itr)   # the whole serialized request
    (fn, args, kwargs) = pickle.loads(payload)                # payload is dead from here
    ...
    result = fn(self, *args, **kwargs)                        # the entire SPU computation
    response = cloudpickle.dumps(result)                      # a second full copy
    for split in split_message(response):                     # CHUNK_SIZE = 10 MiB slices
        yield distributed_pb2.RunResponse(data=split)
```

| rank | size | recorder row | reading at the time |
|---|---:|---|---|
| 1 | 4.89 GiB | `std::vector<pybind11::bytes>::emplace_back<std::string>` | C++ vector in libspu |
| 2 | 1.05 GiB | `_PyEval_EvalFrameDefault <- _PyBytes_Resize` | `cloudpickle.dumps` output buffer |
| 3 | 0.45 GiB | `_PyEval_EvalFrameDefault <- PyBytes_FromStringAndSize` | `payload` (later measured at 2,154 B: `off-path/s1_payload_release/WHY.md`) |
