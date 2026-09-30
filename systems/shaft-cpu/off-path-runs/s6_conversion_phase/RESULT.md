# s6_conversion_phase: three releases during model conversion

Three `s6` candidates, n=3 each, on the then-published `s5_embed_chunk`. Off-path. Levers in
`impl/levers/`: `plaintext_free_early_cpu.py`, `plaintext_free_early_trim_cpu.py`,
`onnx_initializer_release_cpu.py`.

## Runs (VmRSS maxima per phase, kB)

```
                                conversion                   inference                    peak
s5_embed_chunk (base)   2,048,432 / 2,055,244 / 2,061,096   2,030,528/2,053,248/2,052,852   2,061,096
s6_plaintext_free_early 2,054,864 / 2,048,264 / 2,059,988   2,126,180/2,091,784/2,105,192   2,105,192  +2.14%
s6_..._early_trim       2,050,560 / 2,050,588 / 2,044,184   2,069,084/2,073,824/2,088,528   2,073,824  +0.62%
s6_onnx_initializer_rel 2,026,728 / 2,059,956 / 2,052,180   2,119,732/2,092,828/2,082,344   2,092,828  +1.54%
```

```
LEVER|plaintext_free_early_cpu|released_plaintext_bytes=433255432
LEVER|plaintext_free_early_trim_cpu|released_plaintext_bytes=433255432   (plus malloc_trim(0))
LEVER|onnx_initializer_release_cpu|released_protobuf_bytes=433247240
```

- Each lever releases ~433 MB; the conversion peak does not fall; the inference peak rises 2 to 4%.
- With `malloc_trim(0)` the pages return to the kernel and the peak still does not fall.
- The freed holes do not fit the int64 share blocks requested next; the forward pass maps new pages.

## Decomposition at `s5`'s peak

`sample_ts_ms=1786959599921 sample_rss_kb=2057864` (same as `peaks/PEAK`):

```
867.0 MB  42.2%  libstdc++.so.6.0.30+0xf4979   the protobuf parse
433.5 MB  21.1%  python3.10+0x13cc27           the ONNX byte buffer
433.0 MB  21.1%  libc10.so+0x4a675             torch tensors
```

- 867.0 MB = 2 x 433 MB: `numpy_helper.to_array` returns a fresh `bytes` for `node.raw_data`, so
  each weight exists inside and outside the protobuf message.
- The 433.0 MB of `libc10` are the tensors `from_onnx` is building, not the plaintext model the
  first two levers release. The third run attacks rank 1 (protobuf) without effect.

## Phases with the allocator pinned (`p_phases_pinned_s5`)

```
conversion  2,015,996 / 2,037,576 / 2,031,412 kB
inference   1,781,740 / 1,790,984 / 1,750,020 kB    12 to 14% lower
```

The ONNX import sets the peak of the whole-model program; the secure computation needs 1.75 to
1.79 GB. The then-published line stood at 2,056,996 kB.

## Instrument findings

| finding | effect |
|---|---|
| Python census inherited `PEAK_TRIGGER_PCT=2` | on a plateau the last census sat at 1,545,736 against a 2,073,824 kB peak (74.5%); with `PEAK_TRIGGER_PCT=0` at 97.9% |
| poll trace columns | field 2 is VmRSS, field 4 is VmHWM (monotone); phase maxima must use field 2 |
