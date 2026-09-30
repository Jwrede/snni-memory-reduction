# p_gpukeys: key-bank structure, GPU against CPU

Job 45847716 plus a corrected follow-up read, 2026-08-06. Source introspection in
`moai-gpu-s0_default.sif`; nothing measured.

Question at the time: MOAI-CPU's key audit measured 30 default Galois keys plus a separate 47-key
bootstrap bank; this system's target then counted 31 + 1 keys and no second bank.

`Bootstrapper.cu`, every `addBootKeys*` variant:

```cpp
void Bootstrapper::addBootKeys_3(PhantomGaloisKey &gal_keys)
{
  vector<int> gal_steps_vector;
  gal_steps_vector.push_back(0);
  for (int i = 0; i < logNh; i++) gal_steps_vector.push_back((1 << i));
  addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
  ckks->decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks->galois_keys));
```

CPU sibling:

```cpp
keygen.create_galois_keys(gal_steps_vector, gal_keys_boot);   // a SECOND, separate object
```

| | default rotation set | bootstrap rotation set | resident |
|---|---|---|---|
| MOAI-CPU | `gal_keys`, 30 keys | `gal_keys_boot`, 47 keys, separate object | both |
| MOAI-GPU | -- | written into `ckks->galois_keys` | one bank |

- The attention path takes its rotations from the same object (`PhantomGaloisKey &RotK` in
  `single_att_block.cuh`).
- Not settled: this system's own key count (bank = `addLeftRotKeys_Linear_to_vector_3` output plus 0
  and the `logNh` powers of two; earlier campaign's markers: 40.28 GiB). Not a license for a CPU lever
  without checking the CPU rotation set.
- Probe defect: `find / -name "*.cu" -path "*moai*" | head -1` found
  `/moai/build/CMakeFiles/3.22.1/CompilerIdCUDA` and `[ -d "$R" ]` passed; the guard is now
  `test -f $R/test.cu`.
