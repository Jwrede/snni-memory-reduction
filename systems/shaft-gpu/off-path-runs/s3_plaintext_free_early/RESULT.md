# s3_plaintext_free_early: not a step

Jobs 46146787-89, `gpuh200`, n=3, rc=0. Earlier ordering, base `s2_embed_chunk`. Port of
SHAFT-CPU's `plaintext_free_early_cpu.py`: release the caller's plaintext weights before
`from_onnx` parses the ONNX buffer.

```
s2_embed_chunk run1   poller max RSS 2,619,084 kB at ts 1785700451208   (t = 22.9 s)
                      marker after_model_cuda      at ts 1785700458614   (t = 30.3 s)
```

```
plaintext_free_early_gpu: patched
LEVER|plaintext_free_early_gpu|released_plaintext_bytes=433255432
```

```
s2_embed_chunk           min 2,936,380  max 3,130,440  median 2,939,576  spread 6.595%
s3_plaintext_free_early  min 2,935,312  max 2,941,656                    spread 0.216%
```

- The peak is unchanged; the spread falls from 6.595% to 0.216%.
- The published host peak is a spike the 100 ms poller misses: `host peak 2942144 kB is the
  kernel's VmHWM; the 100 ms poller only observed 2858424 kB (2.8% low)`. Object tables and the
  Python census fire on the poller's maximum, so they describe an instant below the published peak.
- Gate failure inherited from `s2_embed_chunk` (phase B).
- Open at the time: a sampler on allocation rather than on a timer.
