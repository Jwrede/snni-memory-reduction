# p_keyaudit: key audit

Job 45846530, 2026-08-06, 13 min 33 s on `long`, probe image `moai-cpu-p_keyaudit.sif`, stopped after
both key sets were generated.

```
POOL|before_gal_keys|alloc_bytes=4902630792        MEM|rss_kb=4994052
KEYAUDIT|gal_keys|keys=30|save_bytes=39636815794
POOL|after_gal_keys|alloc_bytes=46284682632        MEM|rss_kb=43714216
KEYAUDIT|gal_steps_vector|requested=47
KEYAUDIT|gal_keys_boot|keys=47|save_bytes=62097380949
POOL|after_gal_keys_boot|alloc_bytes=112124245400  MEM|rss_kb=104469540
```

| | bytes per key-switching key |
|---|---:|
| derived in MANIFEST.md from the modulus chain | 1,321,205,760 |
| measured, `gal_keys` (39,636,815,794 / 30) | 1,321,227,193 |
| measured, `gal_keys_boot` (62,097,380,949 / 47) | 1,321,220,871 |

- 0.0016% apart (serialisation headers).
- Default set: 30 keys (31 default steps; +16,384 and -16,384 map to one Galois element). Bootstrap
  set: 47 keys, a second object.

Key banks and hidden state without streaming (the streamed target is MANIFEST.md):

```
78 key-switching keys (30 gal + 47 boot + 1 relin) x 1,321,205,760 B  = 100,638,720 kB   95.98 GiB
hidden state, 768 ciphertexts at 21 limbs                             =  16,515,072 kB   15.75 GiB
                                                                        ---------------------------
total                                                                    117,153,792 kB  111.73 GiB
```

- Setup ends at ~100 GiB (pool 4.90 -> 112.12 GB, RSS 104.47 GB), live key bank; the remaining ~284 GiB
  of the 384.29 GiB peak accumulates in the layers.
- Probe defect: a direct `apptainer exec ... /root/moai/build/test` hung; `read_input()` opens weights
  by relative path and the runner builds the working directory with symlinks to `data/`. Without it
  the matrices stay zero silently. The probe now uses `run_moai_cpu.sh`.
