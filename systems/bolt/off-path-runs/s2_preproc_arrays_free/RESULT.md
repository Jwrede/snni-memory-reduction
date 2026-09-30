# s2_preproc_arrays_free: self-derived free lever, off the closed arm

Jobs 46162452-54, n=3, rc=0, on `s1_stream_weights` at `OMP_NUM_THREADS=16`.

```
s1_stream_weights      17,502,624 kB   wall 673 s
s2_preproc_arrays_free 16,685,984      wall 571 s   -4.67%, spread 0.340%, and 15% FASTER
runs                   16,648,724 / 16,705,384 / 16,685,984
```

- Attacks rank 1 of `s1`'s table (`linear.cpp:223:Linear::params_preprocessing_ct_pt`, 4,841.1 MB,
  27.2%); the row falls by its whole size.
- `patch_preproc_arrays_free.py`: "DONOR. None ... found by reading the function named by the
  recorder at rank 1." The closed arm admits only borrowed levers.
- Was first published on the line by oversight, also as a free step behind the paid `s1` (rule 6).
  `steps/s2_threads_4` runs `s1`'s binary (`18a9f64f`); this image is `ef55363d`.

```
preproc_arrays_free   self-derived, FREE, -4.67% and 15% faster
threads_4             borrowed,     PAID, -2.0%  and 18% slower
```
