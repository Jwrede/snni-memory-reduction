# p_depth10: recorder at ten frames

Not a step: one layer, 90 GB, `express`, OOM-killed (`rc=137`) after 9 min 49 s during key generation.
Peak meaningless; names are the measurement.

```
job 46377184   express r04n06   2026-08-22T14:05:23Z
pm_depth_requested=10   pm_depth_max=12   pm_depth=10     <- the first run in this campaign
                                                             that got the depth it asked for
MOAI_N_LAYERS=1   image moai-cpu-m0_default.sif   polled_peak_kb=82,931,272   rc=137
```

Question: at effective depth 6, 91.5% of the peak sat in `Ciphertext::operator=` (58.0%) and
`Ciphertext::resize` (33.5%); does more depth name program lines? (Until 2026-08-22 `pmrec` capped
depth at 6 silently.)

Derive (job 46377185, campaign resolver):

```
39,456.1 MB  49.1%  heap:test_full_scheme.hpp:465:all_layer_test()   <- MemoryPoolHeadMT::get()
35,892.0 MB  44.7%  heap:test_full_scheme.hpp:509:all_layer_test()   <- MemoryPoolHeadMT::get()
 1,459.1 MB   1.8%  heap:seal::SEALContext::create_next_context_data
 1,260.0 MB   1.6%  heap:keygenerator.h:99:seal::KeyGenerator::create_relin_keys(RelinKeys&)
 1,126.1 MB   1.4%  heap:test_full_scheme.hpp:454:all_layer_test()
   299.8 MB   0.4%  anon:[heap]
```

```
465   GaloisKeys gal_keys_boot;                        the bootstrap key set
509   bootstrapper.slot_vec.push_back(logn);           the bootstrapper's own setup
454   cout << "Set encryption parameters and print"    the SEALContext construction folded in
```

- 93.8% in two program lines (line 465 is a declaration: outermost inline frame).
- `peek_live.py` on the same process showed `seal::MemoryManager::GetPool()` (`memorymanager.h:516`)
  33,445.4 MB (87%): `addr2line -i` prints innermost first, `symbolise.resolve(inline=True)` returns the
  outermost. Full chain: `GetPool() -> KSwitchKeys::KSwitchKeys() -> GaloisKeys::GaloisKeys() ->
  all_layer_test()  test_full_scheme.hpp:465` (`impl/peek_deep.sh`).
- Measured during key generation (82.9 GB), not at the 402 GB layer-phase peak.
