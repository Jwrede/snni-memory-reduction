# m1_gelu_release: withdrawn before running

Declared 2026-08-25, built (image sha256 9df062b2, from `p_prm0`), queued as job 46420837, cancelled
the same day. `levers.env` beside this file, unedited. No measurement.

Would have released each power of x inside `gelu_v2` after the summation loop read it:

```
for (int i = 1; i < 25; ++i){
  ...
  evaluator.add_inplace(res, x_n[i]);
  x_n[i] = Ciphertext();          <- the lever
}
```

m0's rank 1:

```
234,877.6 MB  0.570  heap:gelu_others.hpp:143:gelu_v2(...)  <- seal::Ciphertext::operator=
```

The assignment is at `test_full_scheme.hpp:976`:

```
gelu_output[i*32+j] = gelu_v2(inter_output[i*32+j], context, relin_keys, secret_key);
```

`--program-root` reports the first frame under the program; inlining puts `gelu_v2`'s line there. The
memory belongs to `gelu_output` (3,072 elements); `x_n` is local to `gelu_v2`.

| check | result |
|---|---|
| unit test of `tools/attrib_seal_pool/` against the baseline's patched SEAL (`Ciphertext::resize` -> `DynArray::resize` -> `allocate` -> `head->get()`): 25 rounds of 400 ciphertexts at 16 threads, 10,000 takes and releases | table returns to `live=151 bytes=15309784` after every round; `forget()` works under concurrency; no stale attribution |
| source | `gelu_output` filled at line 976, last read at 1019, never released; in scope through `after_bootstrap4` (the peak) |

Replaced by `steps.pending/m1_gelu_output_release/` (`vector<Ciphertext>().swap(gelu_output);` after
the last read; six other vectors in the file are released that way), measured in
`../m1_gelu_output_release/`.
