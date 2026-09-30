# x4f_seal_pool_new: SEAL per-object pools on b3

| | |
|---|---|
| jobs | 47061607-09 (runs), 47061610-12 (derives), 2026-09-30 |
| machine | PALMA 36-core Skylake, 95 GB; `BB_MAX_CONCURRENCY=16`, `OMP_NUM_THREADS=16` |
| image | `bumblebee-f_b4_seal_pool_new.sif` (digest in `RUN_META`), from `impl/tree/`: `b3_weight_defer` with faithful operators plus `seal::MemoryManager::SwitchProfile(std::make_unique<seal::MMProfNew>())` |
| note | the comment beside that line in `op_bench.cc` quotes the placeholder line's numbers |

| | b3_weight_defer | x4f (MMProfNew) | change |
|---|---:|---:|---:|
| peak_host | 6,962,708 kB | 7,966,316 kB | +14.4% |
| replicate spread | 0.15% | 1.21% | |
| wall, median corrected | 346.0 s | 365.4 s | +5.6% |

Each SEAL object gets its own pool head; after `b1`/`b2` removed the mass materialisation, many
private pools hold more than one shared pool. Placeholder line: +12.7%.
