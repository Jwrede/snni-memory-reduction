#!/usr/bin/env python3
"""Returns large pool blocks to the OS instead of SEAL's free list.
Usage: patch_pool_threshold.py <path to the moai source root>

Target: sealpool:retained (46.9% of m1_minimal_att_keys's peak). Items >= 4 MiB (compiled in;
SNNI_POOL_THRESHOLD overrides, 0 = parent policy) freed on return; the recorder keeps recording.
"""
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
ROOT = sys.argv[1]
SEAL = os.path.join(ROOT, "thirdparty", "SEAL-4.1-bs", "native", "src", "seal")
MEMPOOL_H = os.path.join(SEAL, "util", "mempool.h")
MEMPOOL_C = os.path.join(SEAL, "util", "mempool.cpp")

for p in (MEMPOOL_H, MEMPOOL_C):
    if not os.path.isfile(p):
        sys.exit(f"FAILED: {p} is not where this patch expects it")

# Parent must already carry the recorder; against a stock tree this anchor is absent.
_h = open(MEMPOOL_H, encoding="utf-8").read()
if "class MemoryPoolRecordingMT" not in _h:
    sys.exit("FAILED: this tree has no MemoryPoolRecordingMT; apply patch_recording_pool.py first")


def edit(path, old, new, what, already, count=1):
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


# 1. The head that frees on return, and still records (item_count_ kept for alloc_byte_count()).
HEAD = r"""
        // SNNI: a head that returns its block to the OS instead of caching it, and still records.
        // Same recording contract as MemoryPoolHeadRecordingMT: note on get, forget on add. The
        // ONLY difference from it is that `add` frees rather than pushing onto the free list.
        class MemoryPoolHeadRecordingImmediate : public MemoryPoolHead
        {
        public:
            MemoryPoolHeadRecordingImmediate(std::size_t item_byte_count);

            ~MemoryPoolHeadRecordingImmediate() noexcept override
            {}

            SEAL_NODISCARD inline std::size_t item_byte_count() const noexcept override
            {
                return item_byte_count_;
            }

            SEAL_NODISCARD inline std::size_t item_count() const noexcept override
            {
                return item_count_.load(std::memory_order_relaxed);
            }

            MemoryPoolItem *get() override;

            void add(MemoryPoolItem *new_first) noexcept override;

        private:
            MemoryPoolHeadRecordingImmediate(const MemoryPoolHeadRecordingImmediate &copy) = delete;

            MemoryPoolHeadRecordingImmediate &operator=(
                const MemoryPoolHeadRecordingImmediate &assign) = delete;

            const std::size_t item_byte_count_;

            std::atomic<std::size_t> item_count_;
        };

        // SNNI: the step's value, compiled in. The earlier campaign's threshold for this same SEAL
        // fork, and at this parameter set it catches every ciphertext (21.7 MB at bootstrap output,
        // 37.75 MB at full level) and every key element (1,321,205,760 B) while leaving NTT tables
        // and small scratch cached, where returning them to the kernel would only cost syscalls.
#ifndef SNNI_POOL_THRESHOLD_DEFAULT
#define SNNI_POOL_THRESHOLD_DEFAULT (std::size_t(4) * 1024 * 1024)
#endif

        // Read once, cached. `SNNI_POOL_THRESHOLD` overrides it; 0 gives the parent policy back.
        std::size_t snni_pool_threshold();

        class MemoryPoolRecordingMT : public MemoryPoolMT"""
edit(
    MEMPOOL_H,
    "\n        class MemoryPoolRecordingMT : public MemoryPoolMT",
    HEAD,
    "MemoryPoolHeadRecordingImmediate + threshold accessor (mempool.h)",
    already="class MemoryPoolHeadRecordingImmediate",
)

# 2. The implementation, and the one line of policy.
IMPL = r"""
        // SNNI: the declared knob, WITH THE STEP'S VALUE COMPILED IN.
        //
        // The default is 4 MiB and not "off", and that is deliberate. An env-only lever fires only
        // when the submitting shell happens to export it, and `--export=ALL` then carries it in
        // silently -- the exact shape that produced a complete, plausible MOAI-CPU measurement
        // with no object table on 2026-08-25 and cost a ten-hour run. Here the IMAGE is the lever:
        // `strings` proves it, the def checks for it, and no environment is required to reproduce
        // the measurement.
        //
        // `SNNI_POOL_THRESHOLD` still overrides it, which is what `runtime_knob: yes` declares: the
        // row's size is a documented function of ONE parameter, and a probe can sweep it without a
        // rebuild. 0 turns the lever off and gives the parent's policy back exactly.
        //
        // Read once and cached, so the hot path is a load. An unparseable value keeps the compiled
        // default and says so, rather than silently becoming 1 and turning every allocation into an
        // mmap -- that would read as a memory result instead of as a misread variable.
        std::size_t snni_pool_threshold()
        {
            static const std::size_t t = []() -> std::size_t {
                std::size_t v = SNNI_POOL_THRESHOLD_DEFAULT;
                const char *e = std::getenv("SNNI_POOL_THRESHOLD");
                const char *src = "compiled";
                if (e && e[0])
                {
                    char *endp = nullptr;
                    unsigned long long parsed = std::strtoull(e, &endp, 10);
                    if (endp == e || (endp && *endp))
                    {
                        std::fprintf(stderr, "POOLTHRESH|unparseable|keeping compiled default\n");
                    }
                    else
                    {
                        v = static_cast<std::size_t>(parsed);
                        src = "env";
                    }
                }
                std::fprintf(stderr, "LEVER|pool_threshold|bytes=%llu|source=%s\n",
                             static_cast<unsigned long long>(v), src);
                std::fflush(stderr);
                return v;
            }();
            return t;
        }

        // The body is the earlier campaign's MemoryPoolHeadImmediate, unchanged except for the two
        // recorder calls. Keeping it identical is deliberate: that version built and ran against
        // this same SEAL fork, so anything that goes wrong here is the recording, not the freeing.
        MemoryPoolHeadRecordingImmediate::MemoryPoolHeadRecordingImmediate(size_t item_byte_count)
            : item_byte_count_(item_byte_count), item_count_(0)
        {
            if (item_byte_count_ == 0 || item_byte_count_ > MemoryPool::max_batch_alloc_byte_count)
            {
                throw invalid_argument("invalid allocation size");
            }
        }

        MemoryPoolItem *MemoryPoolHeadRecordingImmediate::get()
        {
            seal_byte *data = SEAL_MALLOC(item_byte_count_);
            if (!data)
            {
                throw bad_alloc();
            }
            item_count_.fetch_add(1, memory_order_relaxed);
            // The DATA pointer, not the item header, exactly as MemoryPoolHeadRecordingMT does:
            // pmsample counts present pages over the allocation's own address range.
            snni_poolrec::note(data, item_byte_count_);
            return new MemoryPoolItem(data);
        }

        void MemoryPoolHeadRecordingImmediate::add(MemoryPoolItem *new_first) noexcept
        {
            if (new_first)
            {
                snni_poolrec::forget(new_first->data());
                item_count_.fetch_sub(1, memory_order_relaxed);
                SEAL_FREE(new_first->data());
                delete new_first;
            }
        }

        size_t MemoryPoolMT::alloc_byte_count() const"""
edit(
    MEMPOOL_C,
    "\n        size_t MemoryPoolMT::alloc_byte_count() const",
    IMPL,
    "MemoryPoolHeadRecordingImmediate + snni_pool_threshold (mempool.cpp)",
    already="MemoryPoolHeadRecordingImmediate::get",
)

# THE LEVER ITSELF: one line, at the point where a new size class first appears.
edit(
    MEMPOOL_C,
    "            // THE ONE LINE THAT DIFFERS FROM MemoryPoolMT: the head that records.\n"
    "            MemoryPoolHead *new_head = new MemoryPoolHeadRecordingMT("
    "byte_count, clear_on_destruction_);",
    "            // THE ONE LINE THAT DIFFERS FROM MemoryPoolMT: the head that records.\n"
    "            // SNNI LEVER: at or above the declared threshold the head frees on return\n"
    "            // instead of caching. Below it, and with the threshold unset, this is exactly\n"
    "            // the line the parent image carried.\n"
    "            const std::size_t snni_t = snni_pool_threshold();\n"
    "            MemoryPoolHead *new_head = (snni_t && byte_count >= snni_t)\n"
    "                ? static_cast<MemoryPoolHead *>("
    "new MemoryPoolHeadRecordingImmediate(byte_count))\n"
    "                : static_cast<MemoryPoolHead *>("
    "new MemoryPoolHeadRecordingMT(byte_count, clear_on_destruction_));",
    "the threshold branch in MemoryPoolRecordingMT::get_for_byte_count",
    already="MemoryPoolHeadRecordingImmediate(byte_count)",
)

print("patch_pool_threshold.py: done")
