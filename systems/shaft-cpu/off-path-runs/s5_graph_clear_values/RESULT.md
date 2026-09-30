# s5_graph_clear_values: refuted

| | |
|---|---|
| lever | `graph_clear_values_cpu`: drop each value in CrypTen's `Graph.forward` dict after its last consumer |
| chosen from | owner census at `s4`'s peak: five largest groups 70% of the peak, weights held several times |
| predicted | -17% to -30% |
| measured | -0.02% with the allocator pinned (spread 0.259%) |
| disposition | off-path |
| cluster evidence | `/scratch/tmp/j_wred02/snni_campaign/shaft-cpu/results/{s5_graph_clear_values,p_alloc_s4,p_alloc_s5}` |

## First attempt: undecidable

```
s4  leader peaks  2,126,476 / 2,293,620 / 2,340,856   spread  9.35%
s5  leader peaks  2,079,588 / 2,321,548 / 2,080,024   spread 11.63%
```

Sign depends on the measure: leader -9.3%, max over parties +1.2%, min over parties -2.3%. Both
triples are bimodal.

## Cause of the bimodality

- `LEVER|encrypt_spill_cpu|spilled_bytes=866494480|params=201|max_one_share=178151424|loaded=201`,
  identical in all runs and both parties.
- RSS at `after_model_encrypt` over the 12 party-runs of `s4` and `s5` takes one of two values, gap
  ~382 MB:

```
low   1,020,780  1,023,200  1,023,348  1,024,852  1,025,748  1,025,992
high  1,370,224  1,404,700  1,407,168  1,409,572  1,411,420
```

- glibc's dynamic mmap threshold: freeing an mmap'd block raises the threshold to its size; later
  same-size requests come from the arena. The lever frees 201 blocks (largest 178,151,424 B);
  whether the adaptation has happened by the peak is a race.
- Same size as `param_stream_cpu`'s `anon:[heap]` rise (383.6 -> 555.3 MB).

## Second attempt: allocator pinned

`MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=0 MALLOC_MMAP_THRESHOLD_=131072`; the only difference
between the two runs is `graph_clear_values_cpu` (`LEVER|graph_clear_values_cpu|patched edits=3`).

```
                 leader peaks (kB)                     median      spread   wall
p_alloc_s4   2,018,920 / 2,027,692 / 2,030,560       2,027,692     0.574%   130-134 s
p_alloc_s5   2,027,572 / 2,022,324 / 2,027,264       2,027,264     0.259%   130-131 s
                                                        -0.02%
```

Scope: at the pinned operating point the peak is a different peak (2,027,692 against
2,289,520 kB). The duplicated weights in the census are not refuted; this lever does not reach them.

## Allocator pinning itself

```
s4 at the line's operating point   2,289,520 kB   spread 9.35%    wall 105 s
s4 with the allocator pinned       2,027,692      spread 0.574%   wall 131 s
                                     -11.4%       16x tighter     +24%
```

`after_model_encrypt` at 337,912 kB, no bimodality. Not a step (`../p_alloc_s8/RESULT.md`,
Disposition).
