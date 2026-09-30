# s3_preproc_arrays_free: same lever on the old s2_threads_4

n=3, `OMP_NUM_THREADS=4`.

```
s2_threads_4 (old ordering)  17,150,684 kB
this run                     16,293,832      -5.00%, spread 0.107%
runs                         16,293,832 / 16,292,008 / 16,309,436
```

- Off the line: free behind paid (rule 6), and self-derived on the closed arm.
- Re-runs: `off-path-runs/s2_preproc_arrays_free/`, `off-path-runs/s3_threads_4_on_preproc/`.

```
at OMP=4, on threads_4   (this run)   -5.00%
at OMP=16, on s1         (the repair) -4.67%
```

The freed arrays are not a per-worker quantity.
