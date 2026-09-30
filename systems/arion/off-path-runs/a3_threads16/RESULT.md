# a3_threads16: thread step in the superseded ordering

```
a2_gogc50      253,723,168 kB
a3_threads16   214,494,276 kB   -15.46%   arion_threads=16, wall 36,770 s
```

- Paid (+142% wall) with the free `activation_input_release` measured on top of it: inverted order
  (rule 6), refused by the build.
- Its `levers.env` first declared `go:runtime_retained` as the attacked object (an aggregate); the
  build rejected it and the declaration moved to `rlwe.NewElement`.
- Thread step at `a2_gogc50`'s peak: -15.46%; after the free lever (`a4_threads_16` of that ordering):
  -19.77%.
