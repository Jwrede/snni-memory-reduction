#!/usr/bin/env python3
"""Read-only census of SEAL's memory pool (free-list length per size class): live vs retained split
for the single-row peak (MemoryPoolHeadMT::get(), 99.3%). Producer only.

    patch_poolcensus.py <moai-tree>

Image build, before SEAL is configured. Emitted at phase boundaries from the main thread (counting a
free list takes the head's spin lock; not in a signal handler). impl/p_poolcensus.md.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/moai")
seal = root / "thirdparty/SEAL-4.1-bs/native/src/seal"


def edit(path, anchor, addition, already, what):
    """Insert `addition` after `anchor`, idempotently, with the anchor count checked.
    `already` (a string present only after the patch) is passed explicitly, not derived, to avoid
    matching unpatched source and skipping silently.
    """
    p = seal / path
    s = p.read_text()
    if already in s:
        print(f"  already patched: {what}")
        return
    n = s.count(anchor)
    if n != 1:
        sys.exit(f"FAILED: anchor for '{what}' appears {n} times in {path}, expected 1. "
                 f"SEAL moved; re-derive the patch.")
    p.write_text(s.replace(anchor, anchor + addition, 1))
    if already not in p.read_text():
        sys.exit(f"FAILED: applied '{what}' but its marker is absent from {path}")
    print(f"  patched: {what}  ({path})")


# 0. the base declares the census hook, so the walk needs no dynamic_cast and no RTTI ----------
edit(
    "util/mempool.h",
    """            // Return item back to this pool
            virtual void add(MemoryPoolItem *new_first) noexcept = 0;
""",
    """
            // snni_poolcensus: items of this class currently on the free list. Default 0 so only
            // the head this campaign actually runs (the MT one) has to implement it, and so the
            // walk below needs neither a cast nor RTTI.
            SEAL_NODISCARD virtual std::size_t snni_free_count() const noexcept
            {
                return 0;
            }
""",
    "snni_free_count",
    "MemoryPoolHead declares the census hook",
)

# 1. the head reports how many of its items are on the free list ------------------------------
# Walking first_item_ needs the head's own `locked_` spin lock, or the chain can be rewritten under
# the walk.
edit(
    "util/mempool.h",
    # anchored on the MT head's own add(): item_count() is identical in the ST head.
    """            inline void add(MemoryPoolItem *new_first) noexcept override
            {
                bool expected = false;
                while (!locked_.compare_exchange_strong(expected, true, std::memory_order_acquire))
                {
                    expected = false;
                }
                MemoryPoolItem *old_first = first_item_;
                new_first->next() = old_first;
                first_item_ = new_first;
                locked_.store(false, std::memory_order_release);
            }
""",
    """
            // snni_poolcensus: how many of this head's items are currently on the free list, i.e.
            // held by the pool and not handed out. Takes the head's own spin lock, exactly as
            // add() does, because the chain is rewritten without it.
            SEAL_NODISCARD std::size_t snni_free_count() const noexcept override
            {
                bool expected = false;
                while (!locked_.compare_exchange_strong(expected, true, std::memory_order_acquire))
                {
                    expected = false;
                }
                std::size_t n = 0;
                for (MemoryPoolItem *it = first_item_; it != nullptr; it = it->next())
                {
                    n++;
                }
                locked_.store(false, std::memory_order_release);
                return n;
            }
""",
    "snni_free_count() const noexcept override",
    "MemoryPoolHeadMT reports its free-list length",
)

# 2. the pool walks its heads -------------------------------------------------------------------
# pools_ is protected so the walk lives in the class, under pool_count()'s read lock. Returns
# (item_byte_count, item_count, free_count) per head.
edit(
    "util/mempool.h",
    """            SEAL_NODISCARD std::size_t alloc_byte_count() const override;

        protected:
            MemoryPoolMT(const MemoryPoolMT &copy) = delete;
""",
    """
            // snni_poolcensus: one row per size class, read-only, under the existing read lock.
            SEAL_NODISCARD std::vector<std::array<std::size_t, 3>> snni_census() const override
            {
                std::vector<std::array<std::size_t, 3>> out;
                ReaderLock lock(pools_locker_.acquire_read());
                out.reserve(pools_.size());
                for (auto head : pools_)
                {
                    out.push_back({ head->item_byte_count(), head->item_count(),
                                    head->snni_free_count() });
                }
                return out;
            }
""",
    "snni_census() const override",
    "MemoryPoolMT walks its heads",
)

# 2b. the BASE declares the census, so MemoryPoolHandle can forward without a cast -------------
edit(
    "util/mempool.h",
    """            virtual std::size_t alloc_byte_count() const = 0;
""",
    """
            // snni_poolcensus: one row per size class. Default empty, so only the pool this
            // campaign runs (the MT one) implements it and the handle can forward blindly.
            SEAL_NODISCARD virtual std::vector<std::array<std::size_t, 3>> snni_census() const
            {
                return {};
            }
""",
    "snni_census() const\n            {\n                return {};",
    "MemoryPool declares the census",
)

# 3. the header needs <array>, which it does not include today ---------------------------------
edit(
    "util/mempool.h",
    "#include <algorithm>\n",
    "#include <array>\n",
    "#include <array>",
    "mempool.h includes <array>",
)

# 4. the handle forwards, so the program never touches util:: ---------------------------------
edit(
    "memorymanager.h",
    """        SEAL_NODISCARD inline std::size_t alloc_byte_count() const noexcept
        {
            return !pool_ ? std::size_t(0) : pool_->alloc_byte_count();
        }
""",
    """
        // snni_poolcensus: (item_byte_count, item_count, free_count) per size class.
        SEAL_NODISCARD inline std::vector<std::array<std::size_t, 3>> snni_census() const
        {
            return !pool_ ? std::vector<std::array<std::size_t, 3>>{} : pool_->snni_census();
        }
""",
    "snni_census",
    "MemoryPoolHandle forwards the census",
)

# 4b. the pool emits a census whenever it reaches a NEW MAXIMUM -------------------------------
# The pool grows only when a head allocates a new block, so a counter there is the high-water mark.
# Growth (inside get(), holding a spin lock the census also takes) only SETS A FLAG; the census is
# emitted at the top of the next get_for_byte_count, before any lock, one per GiB of growth.
edit(
    "util/mempool.h",
    """        class MemoryPool
        {
        public:
            static constexpr double alloc_size_multiplier = 1.05;
""",
    """
            // snni_poolcensus: high-water tracking, see the patch script for why the emit point
            // is not the detection point.
            static std::atomic<std::size_t> snni_total_;
            static std::atomic<std::size_t> snni_high_;
            static std::atomic<bool> snni_due_;
            static void snni_grew(std::size_t n)
            {
                std::size_t now = snni_total_.fetch_add(n, std::memory_order_relaxed) + n;
                std::size_t hi = snni_high_.load(std::memory_order_relaxed);
                if (now > hi + (std::size_t(1) << 30))
                {
                    snni_high_.store(now, std::memory_order_relaxed);
                    snni_due_.store(true, std::memory_order_relaxed);
                }
            }
""",
    "snni_grew",
    "MemoryPool tracks its own high-water mark",
)

# 4c. the .cpp side: define the counters, count every SEAL_MALLOC, emit before the locks --------
# The footprint grows only at SEAL_MALLOC, so count there; the "there is memory" branch carves from
# an owned block and must NOT count, or the high-water would track handouts rather than growth.
p = root / "thirdparty/SEAL-4.1-bs/native/src/seal/util/mempool.cpp"
src = p.read_text()
if "snni_total_" in src:
    print("  already patched: mempool.cpp counters")
else:
    defs = """
// snni_poolcensus counters, see mempool.h
std::atomic<std::size_t> seal::util::MemoryPool::snni_total_{ 0 };
std::atomic<std::size_t> seal::util::MemoryPool::snni_high_{ 0 };
std::atomic<bool> seal::util::MemoryPool::snni_due_{ false };
"""
    # FOUR SEAL_MALLOC sites, not two (MemoryPoolHeadST has its own copies); all count, since any
    # of them grows the footprint. Counting ST as well cannot under-report.
    hits = src.count("new_alloc.data_ptr = SEAL_MALLOC(")
    if hits != 4:
        sys.exit(f"FAILED: expected 4 SEAL_MALLOC sites in mempool.cpp, found {hits}")
    src = src.replace("new_alloc.data_ptr = SEAL_MALLOC(mul_safe(MemoryPool::first_alloc_count, item_byte_count_));",
                      "new_alloc.data_ptr = SEAL_MALLOC(mul_safe(MemoryPool::first_alloc_count, item_byte_count_));\n"
                      "                MemoryPool::snni_grew(mul_safe(MemoryPool::first_alloc_count, item_byte_count_));")
    src = src.replace("new_alloc.data_ptr = SEAL_MALLOC(new_alloc_byte_count);",
                      "new_alloc.data_ptr = SEAL_MALLOC(new_alloc_byte_count);\n"
                      "                        MemoryPool::snni_grew(new_alloc_byte_count);")
    src = src + defs
    p.write_text(src)
    if src.count("snni_grew(") != 4:
        sys.exit("FAILED: the growth hook did not land at both SEAL_MALLOC sites")
    print("  patched: pool growth counted at both SEAL_MALLOC sites  (util/mempool.cpp)")

# 4d. the emit point: top of get_for_byte_count, before any lock is taken ----------------------
# <iostream> first: mempool.cpp does not include it and the emit writes to std::cerr.
edit(
    "util/mempool.cpp",
    "#include \"seal/util/mempool.h\"\n",
    "#include <iostream>\n",
    "#include <iostream>",
    "mempool.cpp includes <iostream>",
)

edit(
    "util/mempool.cpp",
    """        Pointer<seal_byte> MemoryPoolMT::get_for_byte_count(size_t byte_count)
        {
""",
    """            // snni_poolcensus: a growth band was crossed since the last call, so emit the
            // table HERE, before the reader lock below, and not where the growth was detected --
            // that is inside a head holding its own spin lock, which the census also takes.
            if (snni_due_.exchange(false, std::memory_order_relaxed))
            {
                for (auto &r : snni_census())
                {
                    std::cerr << "SNNI_POOL|at=peak|total_gib="
                              << (snni_total_.load(std::memory_order_relaxed) >> 30)
                              << "|item_bytes=" << r[0] << "|items=" << r[1]
                              << "|free=" << r[2] << std::endl;
                }
            }
""",
    "at=peak",
    "MemoryPoolMT emits the census at each new high-water band",
)

# 5. the program emits the census at every layer boundary --------------------------------------
# Not at the peak (a signal handler taking the spin lock would deadlock); m1 asks how much of the
# pool is held rather than live, which any instant answers.
p = root / "include/test/test_full_scheme.hpp"
src = p.read_text()
anchor = ("    vector<Ciphertext>().swap(rtn);\n\n"
          "    cout <<\"Decrypt + decode result of one layer: \"<<endl;\n")
if "SNNI_POOL|at=layer_end" in src:
    print("  already patched: the layer-boundary census emitter")
else:
    n = src.count(anchor)
    if n != 1:
        sys.exit(f"FAILED: the layer-boundary anchor appears {n} times, expected 1")
    emit = """
    // SNNI_POOLCENSUS: one line per SEAL pool size class at this phase boundary. Read-only. The
    // pool never returns memory to the OS, so item_count x item_bytes is RESIDENT bytes and the
    // free count is the part of it the POOL is holding rather than the program.
    {
        auto snni_rows = seal::MemoryManager::GetPool().snni_census();
        for (auto &snni_r : snni_rows)
        {
            cerr << "SNNI_POOL|at=layer_end|layer=" << layer_id << "|item_bytes=" << snni_r[0]
                 << "|items=" << snni_r[1] << "|free=" << snni_r[2] << endl;
        }
    }
"""
    p.write_text(src.replace(anchor, anchor + emit, 1))
    if "SNNI_POOL|at=layer_end" not in p.read_text():
        sys.exit("FAILED: the emitter is not in the patched file")
    print("  patched: layer-boundary census emitter  (include/test/test_full_scheme.hpp)")

# 6. a SECOND census at the layer START, which is the cheap one -------------------------------
# Layer 0's start census fires right after key generation, so it shows the instrument works and how
# the key bank sits without a 22-hour run; start and end also bracket what a layer added.
anchor2 = '    cout <<"---------------Layer No. "<<layer_id<<"-------------------"<<endl;\n'
src2 = p.read_text()
if "SNNI_POOL|at=layer_start" in src2:
    print("  already patched: the layer-start census emitter")
else:
    n2 = src2.count(anchor2)
    if n2 != 1:
        sys.exit(f"FAILED: the layer-start anchor appears {n2} times, expected 1")
    emit2 = """    {
        auto snni_rows0 = seal::MemoryManager::GetPool().snni_census();
        for (auto &snni_r : snni_rows0)
        {
            cerr << "SNNI_POOL|at=layer_start|layer=" << layer_id << "|item_bytes=" << snni_r[0]
                 << "|items=" << snni_r[1] << "|free=" << snni_r[2] << endl;
        }
    }
"""
    p.write_text(src2.replace(anchor2, anchor2 + emit2, 1))
    if "SNNI_POOL|at=layer_start" not in p.read_text():
        sys.exit("FAILED: the layer-start emitter is not in the patched file")
    print("  patched: layer-start census emitter  (include/test/test_full_scheme.hpp)")

print("  pool census patched: SEAL side and program side, two emit points")
