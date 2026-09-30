# s6_key_window: key file instead of the shared buffer

The dealer writes its key file and the online phase maps it, instead of handing one buffer on.

```
                    peak_host      W       wall    gate
   s5_keybuf_tight  19,052,324 kB  32.45    27 s   COMM:1062390674 KEYS:18075947008
   s6_key_window    18,916,820     32.22    66 s   identical
                        -0.71%                    +169.9%
```

```
   s5_keybuf_tight                                    s6_key_window
   18,387.8 MB  heap:main  <- gpu_mem.cu:77:cpuMalloc    134.2 MB   same row
      453.3 MB  heap:GPUBERT::GPUBERT                    453.3 MB
      226.6 MB  heap:main  <- bert.h:89:GPUBERT          226.6 MB
   ------------------------------------------------------------------
   19,507 MB attributed                              1,253 MB attributed
```

Leader's poller:

```
              s5_keybuf_tight              s6_key_window
   t = 0.0 s  VmRSS 18,918,876 kB          VmRSS 18,916,808 kB   the dealer's buffer, full
   t = 3.5 s        18,918,876                    1,091,092      close() writes the file, frees
   t = 60  s        19,052,052 (rising)     ~1.1 to 1.9 GB       the entire online phase
```

- Object -99.3%; 17.8 GB removed for the whole online phase. `VmHWM` is set by the dealer's fill first.
- Build refusal: `DECOMPOSITION NOT AT THE PEAK s6_key_window: host objects sampled at 2,060,672 kB,
  89.1% below the 18,916,820 kB peak`. Otherwise `object_conformant: yes`, `conformant: yes`, gate
  byte-identical.

## Defect in the first build

Job 46262407 died in 41 s (`gpu_mem.cu:91`, 713 `cudaErrorHostMemoryNotRegistered`): an added
`cpuFree(sigmaKeygen->startPtr, true)` duplicated the free at the end of `close()`:

```cpp
if (keyFile.compare("") != 0) { write; cpuFree(startPtr); }
```

In the stock configuration that branch is dead (empty key file), and the program then does
`sigma->keyBuf = sigmaKeygen->startPtr;`. The guard now fails if the hand-over survives the patch.

## Derived for s6_dealer_window

The dealer only appends (eight sites; `startPtr` read only for `keySize = keyBuf - startPtr` and the
file write), so it can write and rewind per operation:

```
   SNNI_KW|op_high=825125000|ops=97|keysize=18075946656
```

787 MiB against 16.83 GiB (21.9x; mean 178 MiB per operation). A 1 GiB buffer (1.30x margin, backed
by the live `keySize < keyBufSize` assert) removes the dealer's fill as the peak.
