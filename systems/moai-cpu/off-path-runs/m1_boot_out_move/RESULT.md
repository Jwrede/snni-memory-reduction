# m1_boot_out_move: correct move, peak +4.36%

Job 46120485, `bigsmp` r05n01, n=1, rc=0, wall 82,641 s. First-ordering candidate for step 1.

```
m0_default         402,960,964 kB   spread 0.153% over n=3   wall 82,031 s
m1_boot_out_move   420,544,660 kB                            wall 82,641 s
                   +17,583,696 kB   +4.36%                   +0.7%
```

| | |
|---|---|
| marker | `LEVER|boot_out_move|moved_ciphertexts=768` |
| gate | bit-identical to m0 |
| basis | depth-5 table: `seal::Ciphertext::operator=` 212.21 GiB (55.5%), `seal::Ciphertext::resize` 142.21 GiB (37.2%) |
| change | `rtn2` dead after line 1149; `enc_ecd_x_copy` takes its buffer: two 768-ciphertext arrays live where the stock code has three |

- Mechanism of the rise not established. SEAL's `MemoryPoolHeadMT` keeps what it reserved; a move
  changes which buffer is live and when each is released.
- Opposite sign on MOAI-GPU (`p_bootprobe`): an incremental 768 MB release lowered that pool by
  26.9 GB.
- Build gap at the time: with `n_runs=1` a step whose peak rose passed the check
  (`test_conformance.sh` covered `n3_worse` only).
