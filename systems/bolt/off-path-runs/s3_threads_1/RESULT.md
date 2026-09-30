# s3_threads_1: one worker

n=3, rc=0, on `s2_threads_4`. Next rung of the borrowed thread ladder (BumbleBee `b5`-`b8`).

```
s2_threads_4   17,150,684 kB   wall   775 s
s3_threads_1   17,872,768      wall ~1213 s     +722,084 kB   +4.21%   wall +57%
runs           17,848,348 / 17,872,768 / 17,903,948
```

```
                Linear::params_preprocessing_ct_pt      anon:mmap-rest        mappings
s2_threads_4              5,176.8 MB   29.5%            1,877.9 MB   10.7%      138
s3_threads_1              5,278.1 MB   28.8%            2,755.7 MB   15.1%      179
                             +101.3 MB                     +877.8 MB
```

- Named rows unchanged; the regression is unnamed anonymous mapping (rank 3 -> 2).
- Same shape elsewhere:

```
MOAI-GPU  s4_ffn_threads   8 -> 2 workers on `s3`    peak +2.95%   (off-path)
MOAI-GPU  s12_ffn_threads  8 -> 2 workers on `s9`    peak -6.97%   (published)
BOLT      s3_threads_1     4 -> 1 worker  on `s2`    peak +4.21%   (this run)
```

- Not a step (peak rises); the thread ladder ends at 4.
