# p_floor_bench: SHARK target, measured

Target micro-benchmark of the `shark` target (MANIFEST "## Target", thesis Appendix B.1): each
candidate operation at BERT-base dimensions run alone, with the key bytes the party consumes from
the dealer (`KeyBuf` `bytesReceived`) and the call's own resident high point (VmHWM after a
`clear_refs` reset). Source `impl/src/shark_floor_bench.cpp`, procedure `impl/floor_bench.sbatch`.

| | |
|---|---|
| binary | `benchmark-shark_floor` from the campaign image `shark-s0_default` (docker `2e4e40decd2b`, SHARK `6957e5e`, md5 `c899bd2a`); built from `impl/src/shark_floor_bench.cpp` (the image's copy differs from the repo file in comments only) |
| run | `serverc_floor_bench.sh`: the binary with its libraries and the image's loader, on the host (rootless podman on server C cannot load the image); dealer, then party 0 and party 1 with `SHARK_ONESHOT=0` (keys read from disk per call), `OMP_NUM_THREADS=4`; 3 runs |
| machine | server C (jwmaster01), AMD EPYC (virtualized, 16 logical CPUs), 251 GB |
| outputs | `logs/run<i>/floor_party{0,1}.err` (`FLOOR|` lines), `driver.log` (dealer files: server.dat 2,203,582,480 B, client.dat 2,182,348,816 B in every run) |

## The target operation: one 768-wide truncation (`ars::call`, 98,304 elements, f = 16)

```
                 keybytes        per element   RSS before   peak (KiB)   call delta (KiB)
run 1  party 0   215,875,584     2196.0        4,916        257,144      252,228
       party 1   215,875,584     2196.0        5,076        257,272      252,196
run 2  party 0   215,875,584     2196.0        4,928        257,040      252,112
       party 1   215,875,584     2196.0        5,044        257,268      252,224
run 3  party 0   215,875,584     2196.0        4,964        257,144      252,180
       party 1   215,875,584     2196.0        5,076        257,272      252,196
```

- Key bytes: exactly the derived key term, 98,304 x 2196 B = 215,875,584 B, in every run and party.
- Resident: the call holds 252,112 to 252,228 KiB = 258.2 to 258.3 MB at its high point, 18.7 to
  18.8 % above the target of 212,352 KiB (217.4 MB). The target prices the key material at its
  serialized size; in memory each DCF key is a 96-byte structure with three separately allocated
  sub-arrays (MANIFEST, thesis Appendix B.1), so the deserialized keys of the call occupy more than
  the bytes received. That excess is the implementation's representation and counts as waste under
  the target rule.

## The other operations, for reference

```
operation         elements   keybytes       per element   party 0 peak (KiB)
ars_ffnup3072     393,216    863,502,336    2196.0        1,020,812
ars_gelu3072      393,216    892,993,536    2271.0        1,066,008   (shift f+3 = 19)
mul_gelu3072      393,216     37,748,736      96.0        1,068,564
matmul_ffnup      393,216     91,226,112     232.0        1,069,036
matmul_qkv        294,912     69,206,016     234.7        1,069,036
```

The operations run in sequence in one process; glibc keeps the memory each call frees, so from the
second operation on the `RSS before` column includes retention and the per-call deltas understate
those calls. Only the first operation, the target's, is measured from a clean process. The widest
truncation (`ars_ffnup3072`, ~864 MB of keys) and the GELU truncation (~893 MB) are the
unmodified program's per-call width that the target's 768-wide slice replaces.
