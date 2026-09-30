# p_disc: discrimination test for the T1s gate

Jobs 45769864 (one layer) and 45769865 (two layers), 2026-08-04, probe image `llama-online-probe.sif`
(measured overlay plus a depth switch and the reveal probe, same wire trace). README.md makes this
test the admission condition for `T1s`.

```
one layer   party=2 ops=645   bytes=76029168   ordered=136e60045a3ee0d0 commutative=fab4fc47028e84f1
            party=3 ops=658   bytes=131958000  ordered=0a28a10a525ae4d8 commutative=24a5e046bf44ff51
two layers  party=2 ops=1272  bytes=147920848  ordered=91263d55477638d5 commutative=2654e6f97ca10050
            party=3 ops=1297  bytes=260552656  ordered=aeee228fbcad59f2 commutative=8e550f688f4ce983
```

Every field changes with depth.

| | per layer | 1 layer + 11 x per-layer | measured at 12 layers (`../p_wire/`) |
|---|---:|---:|---:|
| party 2, operations | 627 | 7,542 | 7,542 |
| party 2, bytes | 71,891,680 | 866,837,648 | 866,837,648 |
| party 3, operations | 639 | 7,687 | 7,687 |
| party 3, bytes | 128,594,656 | 1,546,499,216 | 1,546,499,216 |

- The trace is affine in the layer count with no residual; probe and measured images (separate
  builds) agree to the byte.
- `T1s` admitted on `ordered`, `commutative` carried beside it.
