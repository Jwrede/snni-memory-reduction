# x_layer_infeed: result

Jobs 45766804 (3 replicates, all channels, through `run_step.sh`), derives 45766805-07, 2026-08-04.

| | baseline s0_default | x_layer_infeed |
|---|---|---|
| host peak, published party | 7,278,356 kB | 7,410,060 kB |
| replicates | n=3, spread 0.331% | n=3, spread 0.344% |
| gate | pass | pass |
| runtime | 193.2 s | 189.1 s (-2.1%) |

Peak +131,088 kB (+1.77%), five times the spread: not a reduction.

```
     5626296 ->      5616068 kB  (-10228)  heap:python3.10+0x13cc27   <- the attacked object
     1098804 ->      1098804 kB       (+0) heap:python3.10+0x1906b8
       43020 ->            0 kB   (-43020) heap:_message.abi3.so+0x35f9e
           0 ->        41076 kB   (+41076) anon:mmap-1
```

- The arena fell 0.18%.
- The binding process is P2, the receiver (`published_party.txt`: pid 1174699, the last of the five
  node services). The receiver must hold all parameters before the SPU computation starts, so the
  split changes transport granularity, not residency.
- An earlier reading (sender-side asynchrony) was dropped: `examples/python/utils/distributed`
  (job 45768177) has no barrier, and the arena is not on the sender.
- What would work: sequence the computation (layer i's parameters consumed and released before layer
  i+1's arrive), twelve SPU calls instead of one. Later measured on SHAFT-CPU and transferred as
  `s1_weight_stream`.
