# p_poolcensus: SEAL pool census by size class (specification)

Written 2026-08-08 from the vendored source; not run. Superseded by the recording SEAL pool
(`tools/attrib_seal_pool/README.md`, `off-path-runs/p_prm0/`).

Problem at the time: the decomposition was one row, `seal::util::MemoryPoolHeadMT::get()`, 99.3% of a
384.29 GiB peak.

`thirdparty/SEAL-4.1-bs/native/src/seal/util/mempool.h`:

```
class MemoryPoolHead                     // one per distinct allocation size
    virtual size_t item_byte_count()     // the size class
    virtual size_t item_count()          // how many items exist in this class
    struct allocation { size_t size; seal_byte *data_ptr; size_t free; ... }

class MemoryPoolMT : public MemoryPool
    size_t pool_count()                  // number of heads = number of size classes
    size_t alloc_byte_count()            // total taken from the OS
    protected: std::vector<MemoryPoolHead *> pools_;
```

| | |
|---|---|
| patch | one public method on `MemoryPoolMT`: walk `pools_` under the read lock, emit per head `item_byte_count()`, `item_count()`, plus `alloc_byte_count()` |
| triggers | the program's phase markers; `SIGUSR1` from the poller at the peak |
| live | `(item_count - free_count) x item_byte_count` (free list walked from `first_item_`) |
| retained | `free_count x item_byte_count` |
| limit | sizes, not owners; small classes aggregated |
| cost | O(size classes + free-list length) per capture; off the allocation path |

Size classes at the pinned parameters:

```
one key-switching key   35 decomp x 2 polys x 36 limbs x 65536 x 8 B  = 1,321,205,760 B
one ciphertext, level 20        2 polys x 21 limbs x 65536 x 8 B      =    22,020,096 B
```
