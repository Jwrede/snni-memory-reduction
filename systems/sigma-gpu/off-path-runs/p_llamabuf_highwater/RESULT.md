# p_llamabuf_highwater: dealer buffer demand

Job 46142898, `gpuh200`, rc=0, 29 s. Probe (reporting only); basis of `s4_dealerbuf_measured`.

After `s3_keybuf_measured` (host peak 22,069,572 kB at the time) rank 1 was 95.0%, exactly 20 GiB:
three pinned `SIGMAKeygen` allocations (`keyBuf` 18 GiB, sized by s3; `llamaBuf1`, `dummyBuf1` 1 GiB
each). `cpuMalloc(size_t, bool pin = true)` -> `cudaHostRegister`: every page resident.

`initDealer` (`backend/sigma.h:302`) gives one party `llamaBuf2`, the other `dummyBuf2`:

```
size_t llamaKeySz = llamaBuf2 - llamaBuf1;   // per layernorm keygen call
memcpy(keyBuf, llamaBuf1, llamaKeySz);
keyBuf += llamaKeySz;
llamaBuf2 = llamaBuf1;                       // reset: llamaBuf holds ONE round
```

`dummyBuf2` is assigned once and never reset (accumulates):

```
llamaBuf1 needs   max(llamaKeySz)
dummyBuf1 needs   sum(llamaKeySz)
```

```
SNNI_BUF|llamabuf_high=281856|llamabuf_alloc=1073741824      275 KiB of 1 GiB   factor 3810
SNNI_BUF|dummybuf_used=6764544|dummybuf_alloc=1073741824    6.45 MiB of 1 GiB   factor  159
```

- `s4_dealerbuf_measured` sizes both to 64 MiB: -8.92% at the time.
- The patch guard first counted mentions of `dummyBuf2` (expected 3; line 302 names it twice); it now
  counts assignments.
- The probe prints `alloc` as the literal `1073741824`; applied before `s4` in the chain, so `s4`'s log
  still reports it while the code allocates 64 MiB. The `used` values are correct.
