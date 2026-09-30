# s4_trace_nograd: refused as out of order

| | |
|---|---|
| jobs | 45520516-21, 3 replicates |
| base | `s3` of the 2026-07-30 ordering |
| peak | 2670192 -> 2501572 kB, -6.3%, free |
| gate | pass, byte-identical |
| verdict | `object_conformant=no` |

```
NON-CONFORMANT OBJECT s4_trace_nograd: attacked 'libc10.so+0x4a675'
but the largest object at the previous peak was 'heap:python3.10+0x13cc27'
```

- `heap:python3.10+0x13cc27` is CPython's allocator holding the in-memory ONNX byte buffer, 867 MB.
- The Python census (`probe_census_cpu.py`) walks `gc.get_objects()` for `torch.Tensor` and does not
  see non-tensor objects. It identifies tensors inside the dominant recorder site `c10::alloc_cpu`.
- Rule: rank with the recorder table, identify inside a row with the census.
- The lever is selectable once a published step's recorder table names its object as largest.
