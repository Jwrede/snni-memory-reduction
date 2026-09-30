# p_floor_bench_boot: MOAI bootstrap instant, measured as one set

Target micro-benchmark of the instant that sets the `moai-cpu` target (MANIFEST "bootstrap, chunk
128", thesis Appendix B.4): the largest bootstrap key stage, the chunk in flight, the relinearization
key and one thread's working set, built as ONE integrated set at T=1, VmRSS from `/proc/self/status`.
Complements `p_floor_bench/` (one key, a 14-key stage) and `p_floor_bench_att/` (the QK^T instant).

| | |
|---|---|
| patcher | `impl/p_floor_bench_boot.py` (patches `test_full_scheme.hpp` right after `create_relin_keys`, before the 30-key and 47-key banks; local encoder/encryptor/evaluator; returns before inference) |
| set | 15 key-switching keys, steps {0, 1, ..., 14} (step 0 = conjugation; the largest CoeffToSlot/SlotToCoeff stage is 14 rotations + 1 conjugation); 128 ciphertexts at 32 limbs; one thread's working set = a modraised ciphertext at 35 limbs + the 12 Paterson-Stockmeyer temporaries of the modular reduction at 32 limbs (as `impl/p_floor_bench.py` builds it); ciphertexts filled by copying a right-sized template |
| build + run | `serverc_floor_boot.sh`: podman image `localhost/moai-stock` (`27ac402e752c`), MOAI `8bcb0ea` with SEAL fork; `git stash` restores the pinned `test_full_scheme.hpp` before the patch (`logs/driver.log`: PATCH_OK, BUILD rc 0, binary md5 `fbf1eea6...`) |
| machine | server C (jwmaster01), AMD EPYC (virtualized, 16 logical CPUs), 251 GB; `SNNI_FLOOR_BOOT=1 OMP_NUM_THREADS=1`; 3 runs, ~9 min each (input read) |

```
read                     VmRSS (kB), run 1
ctx_pub                   7,608,512    context + public key (baseline, before relin)
boot_context              8,991,716    + relin key and the bench's encoder/encryptor/evaluator
boot_after_stage         28,345,408    + 15 stage keys
boot_after_chunk         32,699,020    + 128 ciphertexts at 32 limbs
boot_instant             33,163,924    + one thread's working set (1 x 35 + 12 x 32 limbs)
boot_after_keyswitch     33,180,312    + one rotation of a chunk ciphertext (outside the target)
```

## Against the derivation (MANIFEST)

```
term                 derived kB     measured kB, runs 1 / 2 / 3                  note
15 stage keys        19,353,600     19,353,692 / 19,353,692 / 19,353,692         +0.0005 %
chunk, 128 @ 32      4,194,304      4,353,612 / 4,353,612 / 4,353,644            +3.8 %
relin                1,290,240      1,383,204 / 1,383,204 / 1,383,204            includes the bench's CKKSEncoder/Encryptor/Evaluator
thread, 35 + 12x32   429,056        464,904 / 464,904 / 464,904                  +8.4 %
---------------------------------------------------------------------------------------------
instant              25,267,200     25,555,412 / 25,555,412 / 25,555,444 kB      +1.14 %
                     = 25.87 GB     = 26.17 GB
```

- The key term, 77 % of the target, is exact; a stage key is 1.32 GB resident as in `p_floor_bench/`.
- The ciphertext terms lie above their payload by the SEAL pool retention of building them (encrypt at
  the top level, mod-switch down; the template ciphertexts and their temporaries stay in the pool), the
  same retention `p_floor_bench_att/` reports for its blocks (+0.85 GB there, +0.20 GB here).
- One key switch at T=1 on a 32-limb chunk ciphertext adds 16,388 kB in every run.
- Like the other target micro-benchmarks it confirms the sizing of the terms; it does not test
  minimality.
