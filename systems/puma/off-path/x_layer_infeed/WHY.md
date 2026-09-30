# x_layer_infeed: why

| | |
|---|---|
| census (`impl/p_census.sbatch`, `../p_census_ns2/`) | one site, `heap:python3.10+0x13cc27`, 5,626,296 of 7,281,432 kB (77%) |
| not | any tracked SPU or JAX buffer; does not scale with the layer count (`p_depth6`) |
| reading | serialisation arena of the driver's single call handing the whole parameter pytree to P2 |
| lever | infeed the non-layer remainder plus twelve per-layer subtrees separately; reassemble the pytree inside the SPU computation (same traced computation; only transport changes) |
| assumption tested | the arena is sized by the largest single serialisation |
