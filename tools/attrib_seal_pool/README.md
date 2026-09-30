# attrib_seal_pool: SEAL pool producer

Host-object producer for MOAI-CPU, where SEAL's memory pool sits between the program and `malloc`:
`pmrec` (hooks `malloc`/`free`) sees pool growth, never returns.

At `m0`'s peak, one freeze (`p_prm0`, 2026-08-25):

```
pmrec, pool growth        227.9 GB in    373 items    the top row, 611 MB per item
this producer, live       234.9 GB in 10,818 items    the top row,  21.7 MB per item
```

611 MB is a pool chunk; 21.7 MB is a ciphertext at that modulus level.

## Mechanism

| | |
|---|---|
| patch | `patch_recording_pool.py` adds `MemoryPoolRecordingMT` to SEAL: `MemoryPoolMT`'s policy unchanged (free lists, retention, growth); `get`/`add` overridden on the pool head (`Pointer::release()` calls `head->add(item)`) |
| output | pmrec's table format in a shared mmap |
| readers | `pmsample` (in the poller's freeze, present pages per item via `/proc/<pid>/pagemap`), `resolve_resident.py`, `--program-root`, `smaps_objects.py`, unchanged |
| subtraction | named file-backed mapping, subtracted by `poll_peak.sh` beside `pmtable_`, `devtable_` |
| switches | off unless `SNNI_POOLREC=1`; path `SNNI_POOLREC_TABLE` (`%d` = pid); floor `SNNI_PM_MIN` |

## Cost

```
peak      402,401,204 kB against m0's published 402,436,136      -0.0087%
wall           38,747 s against 38,283 s                          +1.21%
```

## Naming

The recorder inserts two frames between the allocator and its caller; fixed-index names shift
(`Ciphertext::operator=` against `encrypt_zero_symmetric`). Tables from recording-pool runs are
resolved with `--program-root` and compared only with other program-root tables.

## Findings at m0's peak

- 239.1 GB of 402.3 GB held by live objects, 163.2 GB held by the pool for released objects.
- Galois keys: live early (80.2 GB in 2,124 items of 37.75 MB), returned to the pool by the peak.
  Counters: 870,476,903 inserts, 0 drops.

## Files

```
patch_recording_pool.py    the SEAL patch, with its own reasoning in the header
../../systems/moai-cpu/impl/patch_poller_poolrec.py    wires poll_peak.sh (three edits)
../../systems/moai-cpu/impl/wire_poolrec.py            wires measure.sh and the campaign sbatch
../../systems/moai-cpu/impl/make_prm_def.py            derives an instrument image from a step image
```

Wiring is opt-in and inert without `SNNI_POOLREC`.
