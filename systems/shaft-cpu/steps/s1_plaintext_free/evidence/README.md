# s1_plaintext_free: object ranking at this step's peak

Method and cross-check: `steps/s0_default/evidence/README.md`. The recorder resolves 67% of this
peak to `heap:libc10.so+0x4a675` (`c10::alloc_cpu`).

| | |
|---|---|
| run | `results/p_census_s1/run1`, job 45509422 |
| levers | `SHAFT_LEVERS=plaintext_free_cpu,probe_census_cpu` (this step's lever active) |
| party 0 pid | 763738 |

## Effect of the lever

| | s0 | s1 |
|---|---:|---:|
| plaintext float32 weight groups `(768,768)`, `(3072,768)`, `(28996,768)` | present | absent |
| live tensors at the peak | 2589.5 MB | 1723.0 MB |
| measured peak | 3594240 kB | 3059428 kB |

The peak falls 534812 kB for 866500608 B released: the released tensors (mostly 2.36 MB) come from
the glibc arena, and the freed pages are reused by the embedding matmul's temporaries.

## Census at the highest sampled RSS, party 0

```
CENSUS|newmax|rss_kb=2818916|live_tensor_bytes=1723041820|groups=24|allocations=805
      712605696 B  n=4    torch.int64    (28996, 768)
      231211008 B  n=49   torch.int64    (768, 768)
      226492416 B  n=12   torch.int64    (3072, 768)
      226492416 B  n=12   torch.int64    (768, 3072)
      178151424 B  n=6    torch.int64    (1, 128, 28996)
       75497472 B  n=48   torch.float32  (1, 128, 3072)
       39321600 B  n=100  torch.float32  (1, 128, 768)
       18874368 B  n=24   torch.float32  (1, 12, 128, 128)
        9437184 B  n=24   torch.float32  (1, 128, 12, 64)
        3145728 B  n=1    torch.int64    (512, 768)
```

| object | bytes | frac of the 3059428 kB peak | free fix |
|---|---:|---:|---|
| `int64 (28996, 768)` word-embedding shares, 4 copies | 712605696 | 0.233 | yes: two copies are redundant (below) |
| `int64 (768, 768)` attention weight shares, 49 | 231211008 | 0.076 | no: each read by the protocol |
| `int64 (3072, 768)` / `(768, 3072)` FFN weight shares, 12 each | 226492416 each | 0.074 each | no |
| `int64 (1, 128, 28996)` one-hot and mask temporaries, 6 | 178151424 | 0.058 | yes, same mechanism |
| plaintext `float32` tracing leftovers, 196 tensors | 143130624 | 0.047 | probably: activation-shaped, model already encrypted |

## Redundant copies in the embedding's Beaver matmul

The protocol needs three (28996, 768) tensors at the peak (weight share, mask `b`, revealed
`delta`) and holds four:

1. `y - b` in `_arithmetic_function` (`mpc/primitives/arithmetic.py:359,380`): `result = self.clone()`,
   then `result.share` is replaced; the clone is never read.
2. `all_reduce` clones every input (`communicator/distributed_communicator.py:195`), so the revealed
   `delta` coexists with its source share.

## Constraint

A lever that changes the random draws (for example chunking the one-hot matmul, which repartitions
`generate_random_ring_element`) moves the logits past the 1e-6 tolerance. Such a lever is admissible
only as a phase B step (`GATE-TOLERANCE.md`); `s7_embed_chunk` is one.
