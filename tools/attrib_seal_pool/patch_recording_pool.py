#!/usr/bin/env python3
"""MOAI-CPU: SEAL memory pool that records allocations without changing policy.

    patch_recording_pool.py <moai source tree>

Adds MemoryPoolRecordingMT: MemoryPoolMT's policy, plus a record of each item taken/returned, in
pmrec's table format (pmsample, resolve_resident.py, smaps_objects.py unchanged).
Reason: pool returns never reach `free`, so pmrec sees only pool growth.
SEAL change: four friend declarations. Off unless SNNI_POOLREC=1.
"""
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
ROOT = sys.argv[1]
SEAL = os.path.join(ROOT, "thirdparty", "SEAL-4.1-bs", "native", "src", "seal")
POINTER = os.path.join(SEAL, "util", "pointer.h")
MEMPOOL_H = os.path.join(SEAL, "util", "mempool.h")
MEMPOOL_C = os.path.join(SEAL, "util", "mempool.cpp")
HDR = os.path.join(ROOT, "include", "test", "test_full_scheme.hpp")

for p in (POINTER, MEMPOOL_H, MEMPOOL_C, HDR):
    if not os.path.isfile(p):
        sys.exit(f"FAILED: {p} is not where this patch expects it")


def edit(path, old, new, what, already, count=1):
    """Apply one edit, or accept that a previous run already applied it. The `already` marker is
    checked AFTER writing, so a mismatched patch fails here rather than at link or run time."""
    src = open(path, encoding="utf-8").read()
    if already in src:
        print(f"  already present: {what}")
        return
    if src.count(old) != count:
        sys.exit(f"FAILED: '{what}' expected {count} match(es) in {path}, found {src.count(old)}")
    src = src.replace(old, new, count)
    open(path, "w", encoding="utf-8").write(src)
    if already not in src:
        sys.exit(f"FAILED: applied '{what}' but its marker is absent from {path}")
    print(f"  applied: {what}")


# ---------------------------------------------------------------------------------------------
# 1. pointer.h -- friendship: a pool that cannot construct a Pointer is not a pool. Friendship
# isn't inherited, so all four occurrences (Pointer<seal_byte>, Pointer<T>,
# ConstPointer<seal_byte>, ConstPointer<T>) are needed and counted rather than assumed.
edit(
    POINTER,
    "            friend class MemoryPoolMT;",
    "            friend class MemoryPoolMT;\n            friend class MemoryPoolRecordingMT;",
    "friend class MemoryPoolRecordingMT (pointer.h)",
    already="friend class MemoryPoolRecordingMT;",
    count=4,
)

# ---------------------------------------------------------------------------------------------
# 2. mempool.h -- the recorder's interface and the two classes. Overriding get/add on the HEAD
# (not the pool) sees both the take and the return, via Pointer::release() -> head->add(item).
HDR_BLOCK = r"""
        // SNNI pool recorder.
        // Records who takes a pool item and who returns it, into pmrec's own shared-mmap table
        // format, so `pmsample` reads it at the peak inside the poller's freeze and reports
        // RESIDENT bytes. Policy is UNCHANGED: every allocation decision below is MemoryPoolMT's.
        // Off unless SNNI_POOLREC=1 and SNNI_POOLREC_TABLE names a path.
        namespace snni_poolrec
        {
            bool enabled();

            void note(const void *addr, std::size_t bytes);

            void forget(const void *addr) noexcept;
        } // namespace snni_poolrec

        class MemoryPoolHeadRecordingMT : public MemoryPoolHeadMT
        {
        public:
            MemoryPoolHeadRecordingMT(std::size_t item_byte_count, bool clear_on_destruction = false)
                : MemoryPoolHeadMT(item_byte_count, clear_on_destruction)
            {}

            MemoryPoolItem *get() override;

            void add(MemoryPoolItem *new_first) noexcept override;
        };

        class MemoryPoolRecordingMT : public MemoryPoolMT
        {
        public:
            MemoryPoolRecordingMT(bool clear_on_destruction = false) : MemoryPoolMT(clear_on_destruction)
            {}

            SEAL_NODISCARD Pointer<seal_byte> get_for_byte_count(std::size_t byte_count) override;

        private:
            MemoryPoolRecordingMT(const MemoryPoolRecordingMT &copy) = delete;

            MemoryPoolRecordingMT &operator=(const MemoryPoolRecordingMT &assign) = delete;
        };

        class MemoryPoolST : public MemoryPool"""

edit(
    MEMPOOL_H,
    "\n        class MemoryPoolST : public MemoryPool",
    HDR_BLOCK,
    "MemoryPoolRecordingMT + head (mempool.h)",
    already="class MemoryPoolRecordingMT",
)

# ---------------------------------------------------------------------------------------------
# 3. mempool.cpp -- the implementation. Layout mirrors pmrec's field for field (MAGIC_C3, SLOTS
# 1<<17, PM_FRAMES 12), since a divergence would have pmsample read the table as garbage.
# get_for_byte_count is MemoryPoolMT's body with one line changed (the head type); copied rather
# than factored, so the stock path stays untouched.
IMPL = r"""
        // SNNI pool recorder.
        namespace snni_poolrec
        {
            namespace
            {
                // pmrec's table, field for field. `tools/attrib_resident/pmrec.c` is the original
                // and `pmsample.c` the reader; the magic is the contract between them.
                constexpr std::uint64_t PMREC_MAGIC = 0x504d52454333ULL; // PMREC3
                constexpr std::uint32_t PMREC_SLOTS = 1u << 17;
                constexpr int PMREC_FRAMES = 12;

                struct Slot
                {
                    volatile std::uintptr_t ptr; // 0 = empty
                    std::uint64_t size;
                    std::uint64_t site;
                    std::uint64_t caller;
                    std::uint64_t frames[PMREC_FRAMES];
                };

                struct Table
                {
                    std::uint64_t magic;
                    volatile std::uint64_t inserts, drops;
                    std::uint64_t slots;
                    Slot s[];
                };

                Table *g_tab = nullptr;
                std::size_t g_floor = 0;
                bool g_on = false;
                bool g_init = false;

                inline std::uint32_t hsh(std::uintptr_t p)
                {
                    p >>= 4;
                    p *= 0x9E3779B97F4A7C15ULL;
                    return (std::uint32_t)(p >> 32) & (PMREC_SLOTS - 1);
                }

                void init_once()
                {
                    if (g_init)
                    {
                        return;
                    }
                    g_init = true;
                    const char *e = std::getenv("SNNI_POOLREC");
                    if (!e || e[0] != '1')
                    {
                        return;
                    }
                    // THE FLOOR IS THE INSTRUMENT'S, NOT THIS FILE'S. It defaults to the same
                    // SNNI_PM_MIN the host recorder is given, so the two tables of one run can be
                    // read against each other; a floor chosen here would make them incomparable
                    // for no stated reason.
                    const char *f = std::getenv("SNNI_POOLREC_FLOOR");
                    if (!f || !*f)
                    {
                        f = std::getenv("SNNI_PM_MIN");
                    }
                    g_floor = (f && *f) ? std::strtoull(f, nullptr, 10) : 65536ULL;

                    const char *path = std::getenv("SNNI_POOLREC_TABLE");
                    if (!path || !*path)
                    {
                        std::fprintf(stderr, "POOLREC|off|no SNNI_POOLREC_TABLE\n");
                        return;
                    }
                    // `%d` expands to this pid, exactly as pmrec does it, so several measured
                    // processes cannot truncate each other's table.
                    char buf[4096];
                    const char *pct = std::strstr(path, "%d");
                    if (pct)
                    {
                        std::size_t head = (std::size_t)(pct - path);
                        if (head < sizeof(buf) - 32)
                        {
                            std::memcpy(buf, path, head);
                            int n = std::snprintf(
                                buf + head, sizeof(buf) - head, "%d%s", (int)getpid(), pct + 2);
                            if (n > 0 && (std::size_t)n < sizeof(buf) - head)
                            {
                                path = buf;
                            }
                        }
                    }
                    std::size_t bytes = sizeof(Table) + (std::size_t)PMREC_SLOTS * sizeof(Slot);
                    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
                    if (fd < 0)
                    {
                        std::fprintf(stderr, "POOLREC|off|cannot open %s\n", path);
                        return;
                    }
                    if (ftruncate(fd, (off_t)bytes) != 0)
                    {
                        close(fd);
                        std::fprintf(stderr, "POOLREC|off|cannot size %s\n", path);
                        return;
                    }
                    void *m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                    close(fd);
                    if (m == MAP_FAILED)
                    {
                        std::fprintf(stderr, "POOLREC|off|cannot map %s\n", path);
                        return;
                    }
                    g_tab = (Table *)m;
                    g_tab->slots = PMREC_SLOTS;
                    g_tab->magic = PMREC_MAGIC; // written last: the sampler reads it as ready
                    g_on = true;

                    // Warm the unwinder HERE rather than in the hot path: backtrace() loads
                    // libgcc's unwinder and allocates on its first call, and an allocation inside
                    // the pool's own get() is the one place that must not allocate.
                    void *warm[PMREC_FRAMES + 8];
                    (void)backtrace(warm, PMREC_FRAMES + 8);

                    std::fprintf(
                        stderr, "POOLREC|on|table=%s|floor_bytes=%llu\n", path,
                        (unsigned long long)g_floor);
                    std::fflush(stderr);
                }
            } // namespace

            bool enabled()
            {
                init_once();
                return g_on;
            }

            void note(const void *addr, std::size_t bytes)
            {
                if (!enabled() || !addr || bytes < g_floor)
                {
                    return;
                }
                // FRAME LAYOUT, and it mirrors pmrec's rather than inventing one.
                //   bt[0] note   bt[1] the head's get()   bt[2] the Pointer ctor
                // so bt[2] is the first frame that is not this instrument, and the chain above it
                // is what a resolver walks. `site` and `caller` keep pmrec's meaning; the chain is
                // what `--program-root` actually uses, and it is recorded generously, because a
                // frame spent here is a frame the predicate cannot reach.
                void *bt[PMREC_FRAMES + 8];
                int n = backtrace(bt, PMREC_FRAMES + 8);
                if (n < 3)
                {
                    return;
                }
                std::uintptr_t p = (std::uintptr_t)addr;
                std::uint32_t i = hsh(p);
                for (std::uint32_t k = 0; k < 64; k++, i = (i + 1) & (PMREC_SLOTS - 1))
                {
                    std::uintptr_t exp = 0;
                    if (__atomic_compare_exchange_n(
                            &g_tab->s[i].ptr, &exp, p, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                    {
                        g_tab->s[i].size = (std::uint64_t)bytes;
                        g_tab->s[i].site = (std::uint64_t)(std::uintptr_t)bt[2];
                        g_tab->s[i].caller = (n > 3) ? (std::uint64_t)(std::uintptr_t)bt[3] : 0ULL;
                        for (int f = 0; f < PMREC_FRAMES; f++)
                        {
                            g_tab->s[i].frames[f] =
                                (3 + f < n) ? (std::uint64_t)(std::uintptr_t)bt[3 + f] : 0ULL;
                        }
                        __atomic_fetch_add(&g_tab->inserts, 1, __ATOMIC_RELAXED);
                        return;
                    }
                }
                // Never silently lose count: a full table has to be visible as a full table.
                __atomic_fetch_add(&g_tab->drops, 1, __ATOMIC_RELAXED);
            }

            void forget(const void *addr) noexcept
            {
                if (!g_on || !addr)
                {
                    return;
                }
                std::uintptr_t p = (std::uintptr_t)addr;
                std::uint32_t i = hsh(p);
                for (std::uint32_t k = 0; k < 64; k++, i = (i + 1) & (PMREC_SLOTS - 1))
                {
                    if (__atomic_load_n(&g_tab->s[i].ptr, __ATOMIC_ACQUIRE) == p)
                    {
                        g_tab->s[i].size = 0;
                        __atomic_store_n(&g_tab->s[i].ptr, (std::uintptr_t)0, __ATOMIC_RELEASE);
                        return;
                    }
                }
            }
        } // namespace snni_poolrec

        MemoryPoolItem *MemoryPoolHeadRecordingMT::get()
        {
            MemoryPoolItem *item = MemoryPoolHeadMT::get();
            // The DATA pointer, not the item header: pmsample counts present pages over the
            // allocation's own address range, and that range is the data.
            if (item)
            {
                snni_poolrec::note(item->data(), item_byte_count());
            }
            return item;
        }

        void MemoryPoolHeadRecordingMT::add(MemoryPoolItem *new_first) noexcept
        {
            if (new_first)
            {
                snni_poolrec::forget(new_first->data());
            }
            MemoryPoolHeadMT::add(new_first);
        }

        Pointer<seal_byte> MemoryPoolRecordingMT::get_for_byte_count(size_t byte_count)
        {
            if (byte_count > max_single_alloc_byte_count)
            {
                throw invalid_argument("invalid allocation size");
            }
            else if (byte_count == 0)
            {
                return Pointer<seal_byte>();
            }

            ReaderLock reader_lock(pools_locker_.acquire_read());
            size_t start = 0;
            size_t end = pools_.size();
            while (start < end)
            {
                size_t mid = (start + end) / 2;
                MemoryPoolHead *mid_head = pools_[mid];
                size_t mid_byte_count = mid_head->item_byte_count();
                if (byte_count < mid_byte_count)
                {
                    start = mid + 1;
                }
                else if (byte_count > mid_byte_count)
                {
                    end = mid;
                }
                else
                {
                    return Pointer<seal_byte>(mid_head);
                }
            }
            reader_lock.unlock();

            WriterLock writer_lock(pools_locker_.acquire_write());
            start = 0;
            end = pools_.size();
            while (start < end)
            {
                size_t mid = (start + end) / 2;
                MemoryPoolHead *mid_head = pools_[mid];
                size_t mid_byte_count = mid_head->item_byte_count();
                if (byte_count < mid_byte_count)
                {
                    start = mid + 1;
                }
                else if (byte_count > mid_byte_count)
                {
                    end = mid;
                }
                else
                {
                    return Pointer<seal_byte>(mid_head);
                }
            }

            if (pools_.size() >= max_pool_head_count)
            {
                throw runtime_error("maximum pool head count reached");
            }

            // THE ONE LINE THAT DIFFERS FROM MemoryPoolMT: the head that records.
            MemoryPoolHead *new_head = new MemoryPoolHeadRecordingMT(byte_count, clear_on_destruction_);
            if (!pools_.empty())
            {
                pools_.insert(pools_.begin() + static_cast<ptrdiff_t>(start), new_head);
            }
            else
            {
                pools_.emplace_back(new_head);
            }

            return Pointer<seal_byte>(new_head);
        }

        size_t MemoryPoolMT::alloc_byte_count() const"""

edit(
    MEMPOOL_C,
    '#include "seal/util/mempool.h"',
    '#include "seal/util/mempool.h"\n'
    "// SNNI pool recorder\n"
    "#include <execinfo.h>\n"
    "#include <fcntl.h>\n"
    "#include <sys/mman.h>\n"
    "#include <unistd.h>\n"
    "#include <cstdint>\n"
    "#include <cstdio>\n"
    "#include <cstdlib>\n"
    "#include <cstring>",
    "recorder includes (mempool.cpp)",
    # Marker that cannot appear in the inserted implementation below (a prior marker collided
    # with IMPL's own text, so this edit silently no-op'ed and SEAL failed to compile).
    already="#include <execinfo.h>",
)

edit(
    MEMPOOL_C,
    "\n        size_t MemoryPoolMT::alloc_byte_count() const",
    IMPL,
    "recorder + MemoryPoolRecordingMT (mempool.cpp)",
    already="MemoryPoolRecordingMT::get_for_byte_count",
)

# ---------------------------------------------------------------------------------------------
# 4. The measured program: install the pool, inside all_layer_test() where the measured workload
# begins. MMProfFixed is upstream's own facility for a caller-supplied pool. No dump hook here:
# the table is a shared mapping and the poller samples it at the peak.
edit(
    HDR,
    "void all_layer_test(){",
    r"""void all_layer_test(){
    // SNNI pool recorder, opt-in. Default off, and then this block is the only trace of it.
    {
        const char *pr = getenv("SNNI_POOLREC");
        if (pr && pr[0] == '1') {
            MemoryManager::SwitchProfile(std::unique_ptr<MMProf>(new MMProfFixed(
                MemoryPoolHandle(std::make_shared<seal::util::MemoryPoolRecordingMT>()))));
            fprintf(stderr, "POOLREC|profile_installed\n");
            fflush(stderr);
        }
    }""",
    "install MemoryPoolRecordingMT",
    already="POOLREC|profile_installed",
)

print("patch_recording_pool.py: done")
