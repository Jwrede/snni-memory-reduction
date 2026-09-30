# p_wire: determinism of the structural observable

Jobs 45769303, 45769304, 45769305, 2026-08-04: three runs of the measured image
(`llama-online-o0_default.sif`, with the wire trace), unchanged, twelve layers.

| accumulator | definition |
|---|---|
| `ordered` | FNV-1a chain over per-operation hashes (order-sensitive) |
| `commutative` | sum of per-operation hashes |

```
p_wire1a  party=2 ops=7542 bytes=866837648  ordered=3b76e3f1fc87bbc8 commutative=7b6fcf7c3c4fb915
          party=3 ops=7687 bytes=1546499216 ordered=7b74cdd5d35d76c1 commutative=971efce313627d72
p_wire1b  (identical, both lines)
p_wire2   (identical, both lines)
```

- All fields reproduce for both parties: the gate is declared on `ordered`, `commutative` beside it.
- Client (party 3) receives 1.55 GB over 7,687 operations, server (party 2) 867 MB over 7,542.
- `LLAMA_N_LAYERS` is read only by the diagnostic overlay, so `p_wire2` is a third 12-layer replicate.
  Discrimination: `../p_disc/`.
