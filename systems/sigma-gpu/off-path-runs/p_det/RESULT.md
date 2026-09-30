# p_det1..p_det4: determinism check

Jobs 45780793/94 and 45798381/82, four unchanged runs of `bin/sigma-s0_gate3` (linked as
`bin/sigma-s0_default`), 2026-08-04/05.

```
SEEDED|srand=0x5eed|party=0
FILL|nodes=97|weight_tensors=48|weight_elems=56641536|input_elems=98304|input[0]=-4096|input[1]=847
```

```
GATE_NONZERO|0/98304
rc=6
```

- Seed pinned and fill verified (48 weight tensors, 56,641,536 weight elements, `input[0] = -4096` =
  -1.0 at scale 12).
- Output: 98,304 exact zeros; all four runs byte-identical (md5 `9873e3e32b46b6645252d8624ef1b453`)
  because they are zero.
- Same shape as LLAMA online (`systems/llama/off-path-runs/p_reveal/`: one-bit mask, values above bit
  51). SIGMA's `unmaskValues` is not instrumented: hypothesis only.
- Consequence: structural gate `T1s` on `COMM:`/`KEYS:` (`../p_disc/`).

## Sporadic aborts (2026-08-05)

Intermittent abort in `SigmaPeer::recvBytes` on `numRead == toRead` (short receive, no retry). Four
runs with freeze and four without; six completed, three of each, none aborted: the freeze is not
implicated. Rate not established. Fix (retry loop, as in LLAMA) needs a CUDA rebuild; an aborted run
is re-run and recorded. Now carried as
`impl_instrument: patches/inline_01_recvbytes_tolerates_a_signal_interrupted_read.py`.
