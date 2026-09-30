# p_batch_check_only: why

`d1_layer_weights` borrows the online image `shark-s2_layer_weights.sif` (T=4 order), which also
carries the batch-check lever:

```
LEVER|s1_batch_check_stream|per_layer=1|party=0
LEVER|s2_layer_weights|per_layer=1|party=0
```

A step with two mechanisms is published as one only if the second is measured inert.
`batchCheckArithmBuffer` (`common.cpp:22`) belongs to the online parties' authenticated-share check
(from `output::eval`) and does not appear in the dealer baseline's table; absence from the top
objects is not absence of allocation. This run: the dealer measurement against the batch-check
image alone (linked, not copied), 3 replicates, full instrumentation.
