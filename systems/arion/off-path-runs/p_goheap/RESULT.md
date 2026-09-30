# p_goheap: Go heap census

Jobs 45858140 (clean channel, 6 h 29), 45858141 (census channel), 2026-08-07.

| | before | after |
|---|---|---|
| baseline table | `anon:residual` 614,897.5 MB (1.000), `file:arion` 8.8 MB | below |
| `named_frac_host` | 0.0 (Go takes arenas by `mmap` and sub-allocates; the C-allocator recorder sees nothing) | 0.968 |

```
ringqp.Ring.NewPoly        326,687.2 MB   0.561
go:runtime_retained        158,041.5 MB   0.272
rlwe.NewElement             52,429.6 MB   0.090
anon:residual               18,480.9 MB   0.032
structs.Matrix.CopyNew       9,336.5 MB   0.016
```

- 377.7 GiB joined to allocation sites plus 147.2 GiB `go:runtime_retained` = 524.9 GiB of present
  pages in the pagemap snapshot.
- Snapshot 3.2% below the peak (550,292,644 against 568,451,140 kB; limit 10%).

`ringqp.Ring.NewPoly` by caller:

| caller | | |
|---|---:|---:|
| `lintrans.NewLinearTransformation` | 210.6 GiB | 69.1% |
| `rlwe.NewEvaluatorBuffers` | 39.0 GiB | 12.8% |
| `rlwe.NewElementExtended` | 32.1 GiB | 10.5% |
| `rlwe.NewVectorQP` | 22.9 GiB | 7.5% |

The bootstrap's linear-transformation matrices: 210.6 GiB, 38.8% of the peak.

## Failed attempts

| | cause |
|---|---|
| 1 | wrong image (census needs the one with the Go adapter) |
| 2 | `SNNI_HEAPDUMP_PATH` not passed |
| 3 | guard read `stderr.log`, runner writes `markers.txt`; killed a working run |
| 4 | `attrib_go/` not deployed; 570 GB dump could not be joined |

The parser held `addr -> size` for every object (OOM at 8 GB after 480 s) and read the file twice;
both fixed, with a synthetic dump test.

Outcome of this run: `a0_default` peak 568,451,028 kB, `named_frac_host` 0.968, `object_vs_unnamed`
17.68.
