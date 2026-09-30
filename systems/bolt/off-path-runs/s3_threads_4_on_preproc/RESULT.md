# s3_threads_4_on_preproc: thread step on the off-path base

Jobs 46167568-70, n=3, rc=0. `OMP_NUM_THREADS=4` on `s2_preproc_arrays_free` (image `ef55363d`).

```
s2_preproc_arrays_free  16,685,984 kB   wall 571 s
this run                16,277,556      wall 676 s   -2.45%, spread 0.409%, +18.4% wall
runs                    16,322,944 / 16,256,300 / 16,277,556
```

```
old ordering   s1 -> threads_4 -> preproc_arrays_free   ends at 16,293,832 kB
new ordering   s1 -> preproc_arrays_free -> threads_4   ends at 16,277,556 kB
```

- Both orderings end within 0.10% (endpoint spread 0.409%).
- Thread step: -2.0% on `s1`'s binary (published), -2.45% here: the two levers are nearly independent.
