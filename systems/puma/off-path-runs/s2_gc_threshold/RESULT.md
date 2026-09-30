# s2_gc_threshold: remote-object GC threshold

Jobs 46129041-43, 3 replicates, `s1_response_stream_v2`'s image plus `impl/patch_gc_threshold.py`.

- `distributed_impl.py` releases remote objects via `_dec_objref` -> `_garbage_collect` ->
  `builtin_gc` -> `_del_object`, gated on `_GC_COLLECT_THRESHOLD = 50`; PUMA creates only a few
  object refs, so `_garbage_collect` returns immediately. Patch: threshold 1.

```
              peak kB                              wall s
v2        6,381,368 / 6,410,776 / 6,413,628     356 / 365 / 362
gc = 1    6,408,080 / 6,416,192 / 6,395,332     367 / 371 / 369

object 0x2c7866 (the plaintext model)   437,739,520 B in BOTH, to the byte
```

- Lever active (five markers); gate within its control range; wall +1.9%.
- Cause: in `run_on_spu`, `p = ppd.device("P2")(lambda x: x)(params)` is passed into the SPU call, so
  the reference lives until after the computation. `_dec_objref` is not called while the peak stands.
- The plaintext model on P2 is live by construction at the peak. SHAFT-CPU's `s1_plaintext_free`
  does not transfer in this form; dropping it while the driver holds the ref would change what an
  `ObjectRef` means on the SPU side.
