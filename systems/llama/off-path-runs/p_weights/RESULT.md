# p_weights: model weights, refused lever

Jobs 45772905, 45772920, 45772943, 45772950, 45774748 (source introspection in the measured image),
2026-08-04.

Table after `o1_key_stream` of the first ordering (peak 1,545,136 kB):

```
heap:Tensor2D<unsigned long>::Tensor2D(unsigned long, unsigned long)   664,140 kB   43%
```

7,077,888 weights per layer x 12 x 8 B = 679.5 MB: all twelve layers' weights in each `FC`'s
`Tensor2D<u64>`.

## Dealer mechanism (`d5`)

- Shrink each FC weight matrix after initialisation; re-materialise before its matmul
  (`resize(d1,d2)`, `randomize(range)` from `prngWeights`).
- Each `FC` stores `wseed`, `wscale`; `_initScale` seeds from them unconditionally ("Per-layer seeding
  is applied unconditionally, so the eager and lazy runs draw the same values and therefore generate
  byte-identical keys.").

## Why inadmissible online

`llama_base.h:166`, `initializeInferencePartyA`, masks each layer's weights in place:

```cpp
printf("Weight=%ld\n", mha->wQKV.data[0]);
input_layer(mha->wQKV.data, tmp1.data, mha->wQKV.size(), 2);
printf("Masked weight=%ld\n", mha->wQKV.data[0]);
...
auto weights = layer->getweights();      // the same path for every FC layer
```

- At `_forward`, `weight` holds `w + r`; regenerating from `wseed` restores `w`.
- Data-oblivious protocol: wire trace unchanged, output 98,304 zeros regardless (`../p_reveal/`).
- Admissible form: reproduce `r` too (replay or seek the key-file consumption), a change to the
  key-material path. Not attempted.

| system | why an identified object was not a lever |
|---|---|
| PUMA | object on the receiving side; the sender was split (`systems/puma/off-path/x_layer_infeed/`) |
| LLAMA online | object transformed in place between allocation and use |
| BOLT (earlier campaign) | QKV split infeasible |
