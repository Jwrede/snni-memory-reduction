# p_alloc_s8: allocator pinned on `s8`

Operating-point study, not a step. `s8_encrypt_chunk`'s lever set unchanged, allocator pinned, n=3.
Compared with the `s8` runs before the 2026-09-29 re-measurement. Earlier probes of this kind:
`p_alloc_s4`, `p_alloc_s5` (other lever set, peak 2.6x this one).

## Peak on `VmHWM` (the published quantity)

```
                   VmHWM of the three replicates        median   published    spread   wall
s8_encrypt_chunk   845,840 / 885,372 / 921,860         885,372     881,272     8.59%   ~107 s
p_alloc_s8         773,640 / 775,408 / 776,012         775,408     771,308     0.31%    135 s
                                                                    -12.48%      28x     +26%
```

At `s4` the same pinning gave -11.4% and a 16x narrower spread.

## Same runs on the poller maximum

```
                   poller maxima                     median   spread
s8_encrypt_chunk   845,840 / 885,372 / 898,152      885,372    5.91%
p_alloc_s8         653,336 / 653,784 / 715,388      653,784    9.49%
                                                     -26.2%   WIDER
```

`VmHWM` minus poller maximum is about 120,000 kB in the pinned runs: a spike the 100 ms poller does
not see. README.md publishes `VmHWM` for this reason.

## Decomposition at the median run (`make_steps.py` path: `load_objects`, `group_sites`)

```
                        s8 defaults      s8 pinned      change
   anon:residual           645.1 MB        451.1 MB     -194.0
   named heap              122.5            89.1         -33.4
   file: (libraries)       134.8           125.2          -9.7
   instrument                4.2             4.2          +0.0
   the peak itself         902.4           789.8        -112.6
```

```
   s8 defaults                                        s8 pinned
   113.4 MB  heap:torch:torch.int64 held_by=dict      89.1 MB  heap:torch:torch.float32 held_by=OrderedDict
     6.3 MB  heap:torch:torch.int64 held_by=list
     2.4 MB  heap:torch:torch.float32 held_by=OrderedDict
     0.4 MB  heap:torch:torch.float32 held_by=frame
```

- `anon:` falls 1.7x the peak fall; no named object is reduced. Same shape as PUMA `s1_arena_cap`
  (peak -227.4 MB, `anon:` -245.8 MB, named heap -6.7 MB), which is off-path.
- The int64 rows (encrypted shares) and the float32 `OrderedDict` (plaintext state dict) are
  different objects at different instants.

## Peak instant

| | poller maximum |
|---|---|
| defaults | 0.70 s before `layer|after_run_0`, inside encoder layer 0 |
| pinned | 0.80 s after `main|after_input_encrypt`, load-and-encrypt phase |

Pinning removes the allocator's share of the per-layer transient (~545 MB); the deterministic
loading phase becomes the maximum. This also explains the spread collapse: under defaults the peak
falls in one of twelve similar layer transients.

## Disposition

Not a step (decided 2026-08-18): the reduction is in `anon:residual` (71.5% of `s8`'s peak), which
no step may target. Cost would be paid: +26% wall against a 1.90% spread. ARION's `a2_gogc50`
(`runtime_knob: yes`, `object_attacked: go:runtime_retained`) is not a precedent, because a `go:`
row's size is a documented function of a declared parameter and glibc's retention is not.
