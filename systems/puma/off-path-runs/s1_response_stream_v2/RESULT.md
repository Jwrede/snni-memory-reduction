# s1_response_stream_v2: response stream from the baseline

First published as step `s1` (-12.4%); off-path since `s1_weight_stream` became the line.

At the baseline's peak it attacked rank 2 and declared rank 1 skipped:

```
5,528.6 MB  75.6%  pybind11::bytes <- PyBytes_FromStringAndSize      DECLARED SKIPPED
1,125.2 MB  15.0%  python3.10+0x28728a <- _PyBytes_Resize            <- that step
  437.7 MB   5.8%  python3.10+0x2c7866 <- PyBytes_FromStringAndSize
```

The skip rested on three depth probes placing the allocating frame in `libspu.so` (a wheel). The
arena's size follows what the driver asks SPU to hold: `s1_weight_stream` reduces it 53% without C++
changes.

```
response_stream_v2 on s0_default        7,316,112 -> 6,406,676 kB   -12.4%
response_stream_v2 on s1_weight_stream  3,005,148 -> 3,061,920      +1.9%
                                        runs: 3,062,052 / 3,061,920 / 3,011,080
```

Second row: `../s2_response_stream_on_ws/RESULT.md`.
