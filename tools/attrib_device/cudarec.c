/* cudarec.so: device-side CUDA allocation recorder (device counterpart of attrib_resident/pmrec.so).
 * Hooks cudaMalloc(Async) and the _ptsz variants; live bytes per call site; table snapshot at each new
 * peak. Enable: SNNI_CUDAREC=1. Output: $SNNI_CUDAREC_OUT (default /results).
 * Build: attrib_device/build_in_image.sh <image>.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Declared locally so the recorder builds without the CUDA toolkit present. cudaError_t is an
 * enum (int-sized) and cudaStream_t is an opaque pointer; both are ABI-stable. */
typedef int cerr_t;
typedef void *cstream_t;

#define PTR_SLOTS (1u << 20) /* live pointers; open addressing, power of two */
#define MAX_SITES 4096

/* `used`: 0=never occupied, 1=live, 2=tombstone (freed), 3=claimed (thread filling slot).
 * Open addressing requires the tombstone -- clearing to 0 on free reopens holes in probe chains,
 * causing missed frees and stale-pointer reuse that silently corrupts site byte counts. */
struct pent {
    void *ptr;
    size_t sz;
    uint32_t site;
    uint32_t used;
};

struct site {
    void *addr;
    void *caller; /* the frame ABOVE addr, or NULL when depth capture is off */
    void *caller2; /* one frame further, or NULL below SNNI_CUDAREC_DEPTH=3 */
    uint64_t live_bytes;
    uint64_t live_count;
    uint64_t total_allocs;
};

static struct pent *g_ptrs;
static struct site g_sites[MAX_SITES];
static uint64_t g_peak_bytes_by_site[MAX_SITES];
static uint64_t g_peak_count_by_site[MAX_SITES];
static uint32_t g_nsites;

/* Optional second snapshot, taken at the moment the pool's RESERVED memory is highest rather than
 * at the live high. The reported device peak is the pool's reserved high-water, and on a
 * fragmenting allocator that sits at a different instant than the live high, so the default table
 * can decompose a moment the metric does not describe. Opt-in via SNNI_CUDAREC_RESERVED_PEAK=1;
 * default off keeps every other run byte-identical. Reserved is read as device memory in use
 * (total minus free from cudaMemGetInfo), whose high coincides with the pool's reserved high apart
 * from the fixed CUDA context, so no memory-pool attribute enum is hardcoded. */
static uint64_t g_rpeak_bytes_by_site[MAX_SITES];
static uint64_t g_rpeak_count_by_site[MAX_SITES];
static uint64_t g_peak_reserved;    /* max (total - free) device bytes seen */
static uint64_t g_live_at_rpeak;    /* live total captured at that instant */
static int g_rpeak_on;              /* SNNI_CUDAREC_RESERVED_PEAK=1 */
static uint64_t g_rpeak_every = 64; /* query cudaMemGetInfo every Nth alloc, SNNI_CUDAREC_RESERVED_EVERY */
static uint64_t g_rpeak_seq;        /* alloc counter for the throttle */

static uint64_t g_live, g_peak;
/* Hook counters are atomic: incremented from whatever thread made the call. Selftest caught lost
 * increments (a quietly-low count) under concurrent, non-atomic access. */
static uint64_t g_n_malloc_async_ptsz, g_n_malloc_async, g_n_malloc;
static uint64_t g_n_free_async_ptsz, g_n_free_async, g_n_free;
static uint64_t g_drops, g_untracked_frees, g_stale_reuse;

/* Hooks run concurrently from whatever thread made the CUDA call (measured systems are
 * OpenMP-parallel); without synchronisation, slot claims and live_bytes updates race and site
 * balances drift negative. Counters are atomic, slots use compare-exchange; appending a site and
 * snapshotting at a new peak take this spin lock. The allocation fast path stays lock-free. */
static volatile int g_lock;

/* Caller capture, opt-in via SNNI_CUDAREC_DEPTH=2: __builtin_return_address(0) often names an
 * allocation helper rather than the real owner, so backtrace() (safer than
 * __builtin_return_address(1), which can crash without frame pointers) captures one frame
 * further when enabled. Site key becomes the (addr, caller) pair; depth 1 leaves caller NULL. */
static int g_depth = 1;

__attribute__((noinline)) static void *caller_frame(void)
{
    if (g_depth < 2)
        return NULL;
    void *bt[4];
    int n = backtrace(bt, 4);
    /* noinline so the frame index can't depend on optimiser inlining (bit the host recorder before).
       bt[0]=this fn, bt[1]=interposed allocator, bt[2]=call site, bt[3]=frame above (returned). */
    return n >= 4 ? bt[3] : NULL;
}

/* Depth 3, strictly opt-in: on MOAI-GPU both depth-2 frames land inside the FHE library with no
 * owner frame, leaving part of the device peak unattackable, so one more frame out is captured.
 * Below depth 3 this returns NULL and behaviour is byte-identical to before. */
__attribute__((noinline)) static void *caller_frame2(void)
{
    if (g_depth < 3)
        return NULL;
    void *bt[5];
    int n = backtrace(bt, 5);
    /* Same accounting, one frame further: bt[2]=site, bt[3]=its caller, bt[4]=wanted here. */
    return n >= 5 ? bt[4] : NULL;
}

static inline void lock_acquire(void)
{
    int expected = 0;
    while (!__atomic_compare_exchange_n(&g_lock, &expected, 1, 0,
                                        __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
        expected = 0;
}

static inline void lock_release(void)
{
    __atomic_store_n(&g_lock, 0, __ATOMIC_RELEASE);
}
static int g_on;

static cerr_t (*real_malloc_async_ptsz)(void **, size_t, cstream_t);
static cerr_t (*real_malloc_async)(void **, size_t, cstream_t);
static cerr_t (*real_malloc)(void **, size_t);
static cerr_t (*real_free_async_ptsz)(void *, cstream_t);
static cerr_t (*real_free_async)(void *, cstream_t);
static cerr_t (*real_free)(void *);
static cerr_t (*real_meminfo)(size_t *, size_t *);  /* cudaMemGetInfo(free, total) */

static void dump(void);

__attribute__((constructor)) static void init(void)
{
    const char *e = getenv("SNNI_CUDAREC");
    g_on = (e && e[0] == '1');
    const char *d = getenv("SNNI_CUDAREC_DEPTH");
    if (d && d[0]) {
        /* Strict parse+range-check, not pattern match: an unrecognised value must fail loudly
         * rather than silently degrade to a coarser table (this cost a 56-minute GPU run once). */
        char *end = NULL;
        long v = strtol(d, &end, 10);
        if (!end || *end || v < 1 || v > 3) {
            fprintf(stderr, "cudarec: SNNI_CUDAREC_DEPTH=%s is not 1, 2 or 3\n", d);
            _exit(3);
        }
        g_depth = (int)v;
    }

    /* Resolved unconditionally so calls made while the recorder is off still work. Safe here
     * (unlike in pmrec) because these are CUDA symbols, so dlsym's own allocations can't re-enter us. */
    real_malloc_async_ptsz = dlsym(RTLD_NEXT, "cudaMallocAsync_ptsz");
    real_malloc_async = dlsym(RTLD_NEXT, "cudaMallocAsync");
    real_malloc = dlsym(RTLD_NEXT, "cudaMalloc");
    real_free_async_ptsz = dlsym(RTLD_NEXT, "cudaFreeAsync_ptsz");
    real_free_async = dlsym(RTLD_NEXT, "cudaFreeAsync");
    real_free = dlsym(RTLD_NEXT, "cudaFree");
    real_meminfo = dlsym(RTLD_NEXT, "cudaMemGetInfo");

    /* Reserved-peak snapshot mode, opt-in and range-checked like the depth so an unrecognised
     * value fails loudly rather than degrading a 56-minute run. Off by default. */
    {
        const char *rp = getenv("SNNI_CUDAREC_RESERVED_PEAK");
        g_rpeak_on = (rp && rp[0] == '1');
        const char *re = getenv("SNNI_CUDAREC_RESERVED_EVERY");
        if (re && re[0]) {
            char *end = NULL;
            long v = strtol(re, &end, 10);
            if (!end || *end || v < 1) {
                fprintf(stderr, "cudarec: SNNI_CUDAREC_RESERVED_EVERY=%s is not a positive integer\n", re);
                _exit(3);
            }
            g_rpeak_every = (uint64_t)v;
        }
    }

    if (!g_on)
        return;

    /* Pointer table is a named file-backed mapping, not calloc/anonymous heap, so its footprint can
     * be labelled and subtracted from the published peak by smaps_objects.py (an anonymous mapping
     * cannot be). SNNI_CUDAREC_TABLE names the file; without it, falls back to anonymous (old
     * behaviour), logged to stderr either way. */
    {
        const char *tpath = getenv("SNNI_CUDAREC_TABLE");
        size_t tbytes = (size_t)PTR_SLOTS * sizeof *g_ptrs;
        const char *how = "anon";
        char tbuf[4096];
        /* `%d` expands to pid, as with SNNI_PM_TABLE, so co-located parties don't truncate
         * each other's table. */
        if (tpath && strstr(tpath, "%d")) {
            snprintf(tbuf, sizeof tbuf, tpath, (int)getpid());
            tpath = tbuf;
        }
        if (tpath && *tpath) {
            int fd = open(tpath, O_RDWR | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0 && ftruncate(fd, (off_t)tbytes) == 0) {
                void *m = mmap(NULL, tbytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (m != MAP_FAILED) {
                    g_ptrs = m;
                    how = "file";
                }
            }
            if (fd >= 0)
                close(fd);
            if (how[0] != 'f')
                fprintf(stderr, "SNNI_CUDAREC|table_fallback=anon|path=%s\n", tpath);
        }
        if (!g_ptrs)
            g_ptrs = calloc(PTR_SLOTS, sizeof *g_ptrs);
        if (!g_ptrs) {
            fprintf(stderr, "SNNI_CUDAREC|init_failed=cannot allocate pointer table\n");
            g_on = 0;
            return;
        }
        atexit(dump);
        fprintf(stderr, "SNNI_CUDAREC|recording=on|ptr_slots=%u|table=%s|table_bytes=%zu\n",
                PTR_SLOTS, how, tbytes);
    }
}

static inline size_t hash_ptr(void *p)
{
    uint64_t x = (uint64_t)p >> 4;
    x *= 0x9E3779B97F4A7C15ull;
    return (size_t)(x >> 40) & (PTR_SLOTS - 1);
}

/* Lock-free scan: sites are only appended, and g_nsites is read with acquire so a published
 * address is visible. Only append takes the lock, and rescans under it. */
static uint32_t site_index(void *ra, void *caller, void *caller2)
{
    uint32_t n = __atomic_load_n(&g_nsites, __ATOMIC_ACQUIRE);
    for (uint32_t i = 0; i < n; i++)
        if (g_sites[i].addr == ra && g_sites[i].caller == caller && g_sites[i].caller2 == caller2)
            return i;
    lock_acquire();
    for (uint32_t i = 0; i < g_nsites; i++)
        if (g_sites[i].addr == ra && g_sites[i].caller == caller && g_sites[i].caller2 == caller2) {
            lock_release();
            return i;
        }
    if (g_nsites >= MAX_SITES) {
        lock_release();
        return MAX_SITES - 1; /* last slot is the overflow bucket */
    }
    uint32_t idx = g_nsites;
    g_sites[idx].addr = ra;
    g_sites[idx].caller = caller;
    g_sites[idx].caller2 = caller2;
    __atomic_store_n(&g_nsites, idx + 1, __ATOMIC_RELEASE);
    lock_release();
    return idx;
}

/* Called on every allocation. The peak snapshot happens here, which is why it cannot be missed:
 * it is not a sample of the program, it IS the program's own allocation event. */
static void note_alloc(void *p, size_t n, void *ra, void *caller, void *caller2)
{
    if (!g_on || !p)
        return;
    size_t h = hash_ptr(p);
    struct pent *slot = NULL;
    for (size_t i = 0; i < PTR_SLOTS; i++) {
        struct pent *e = &g_ptrs[(h + i) & (PTR_SLOTS - 1)];
        uint32_t st = __atomic_load_n(&e->used, __ATOMIC_ACQUIRE);
        if (st == 1 && e->ptr == p) {
            /* Already live in the table: its free was never seen. Charge the old entry back to
             * its own site before the slot is reused, so stale bytes leave with the site that
             * holds them. Counted, because a recorder that silently repairs itself hides a bug. */
            __atomic_sub_fetch(&g_sites[e->site].live_bytes, e->sz, __ATOMIC_RELAXED);
            __atomic_sub_fetch(&g_sites[e->site].live_count, 1, __ATOMIC_RELAXED);
            __atomic_sub_fetch(&g_live, e->sz, __ATOMIC_RELAXED);
            __atomic_add_fetch(&g_stale_reuse, 1, __ATOMIC_RELAXED);
            slot = e;
            break;
        }
        if (st == 0 || st == 2) {
            /* Claim it. Losing the race means another thread took it; keep probing. */
            uint32_t expected = st;
            if (__atomic_compare_exchange_n(&e->used, &expected, 3, 0,
                                            __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
                slot = e;
                break;
            }
        }
        /* st == 3 is a slot another thread is filling in: not ours, keep probing. */
    }
    if (!slot) {
        __atomic_add_fetch(&g_drops, 1, __ATOMIC_RELAXED);
        return; /* table full: counted and published, never silently discarded */
    }

    uint32_t st = site_index(ra, caller, caller2);
    slot->ptr = p;
    slot->sz = n;
    slot->site = st;
    __atomic_store_n(&slot->used, 1, __ATOMIC_RELEASE);

    __atomic_add_fetch(&g_sites[st].live_bytes, n, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_sites[st].live_count, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_sites[st].total_allocs, 1, __ATOMIC_RELAXED);
    uint64_t live = __atomic_add_fetch(&g_live, n, __ATOMIC_RELAXED);

    /* Snapshot must be consistent as a whole: takes the lock and re-reads live total under it,
     * else one thread could copy sites while another is still updating them. */
    if (live > __atomic_load_n(&g_peak, __ATOMIC_RELAXED)) {
        lock_acquire();
        uint64_t now = __atomic_load_n(&g_live, __ATOMIC_RELAXED);
        if (now > g_peak) {
            g_peak = now;
            uint32_t ns = __atomic_load_n(&g_nsites, __ATOMIC_ACQUIRE);
            for (uint32_t k = 0; k < ns; k++) {
                g_peak_bytes_by_site[k] = __atomic_load_n(&g_sites[k].live_bytes, __ATOMIC_RELAXED);
                g_peak_count_by_site[k] = __atomic_load_n(&g_sites[k].live_count, __ATOMIC_RELAXED);
            }
        }
        lock_release();
    }

    /* Optional second snapshot at the RESERVED high. The reported device peak is the pool's
     * reserved high-water, which on a fragmenting allocator sits at a different instant than the
     * live high above, so the default table can decompose a moment the metric does not describe.
     * Reserved is read as device memory in use (total - free), throttled by SNNI_CUDAREC_RESERVED_EVERY.
     * cudaMemGetInfo allocates nothing, so it cannot re-enter this hook. Off unless opted in. */
    if (g_rpeak_on && real_meminfo &&
        (__atomic_add_fetch(&g_rpeak_seq, 1, __ATOMIC_RELAXED) % g_rpeak_every) == 0) {
        size_t fr = 0, tot = 0;
        if (real_meminfo(&fr, &tot) == 0 && tot >= fr) {
            uint64_t used = (uint64_t)(tot - fr);
            if (used > __atomic_load_n(&g_peak_reserved, __ATOMIC_RELAXED)) {
                lock_acquire();
                if (used > g_peak_reserved) {
                    g_peak_reserved = used;
                    g_live_at_rpeak = __atomic_load_n(&g_live, __ATOMIC_RELAXED);
                    uint32_t ns = __atomic_load_n(&g_nsites, __ATOMIC_ACQUIRE);
                    for (uint32_t k = 0; k < ns; k++) {
                        g_rpeak_bytes_by_site[k] = __atomic_load_n(&g_sites[k].live_bytes, __ATOMIC_RELAXED);
                        g_rpeak_count_by_site[k] = __atomic_load_n(&g_sites[k].live_count, __ATOMIC_RELAXED);
                    }
                }
                lock_release();
            }
        }
    }
}

static void note_free(void *p)
{
    if (!g_on || !p)
        return;
    size_t h = hash_ptr(p);
    for (size_t i = 0; i < PTR_SLOTS; i++) {
        struct pent *e = &g_ptrs[(h + i) & (PTR_SLOTS - 1)];
        uint32_t st = __atomic_load_n(&e->used, __ATOMIC_ACQUIRE);
        if (st == 1 && e->ptr == p) {
            /* Claim the removal, so two frees of the same pointer cannot both decrement. */
            uint32_t expected = 1;
            if (!__atomic_compare_exchange_n(&e->used, &expected, 2, 0,
                                             __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                continue; /* someone else took it; keep probing */
            __atomic_sub_fetch(&g_sites[e->site].live_bytes, e->sz, __ATOMIC_RELAXED);
            __atomic_sub_fetch(&g_sites[e->site].live_count, 1, __ATOMIC_RELAXED);
            __atomic_sub_fetch(&g_live, e->sz, __ATOMIC_RELAXED);
            e->ptr = NULL;
            return;
        }
        if (st == 0)
            break; /* never occupied: nothing was inserted beyond this point */
    }
    __atomic_add_fetch(&g_untracked_frees, 1, __ATOMIC_RELAXED);
}

cerr_t cudaMallocAsync_ptsz(void **p, size_t n, cstream_t s)
{
    cerr_t r = real_malloc_async_ptsz(p, n, s);
    __atomic_add_fetch(&g_n_malloc_async_ptsz, 1, __ATOMIC_RELAXED);
    if (r == 0)
        note_alloc(*p, n, __builtin_return_address(0), caller_frame(), caller_frame2());
    return r;
}

cerr_t cudaMallocAsync(void **p, size_t n, cstream_t s)
{
    cerr_t r = real_malloc_async(p, n, s);
    __atomic_add_fetch(&g_n_malloc_async, 1, __ATOMIC_RELAXED);
    if (r == 0)
        note_alloc(*p, n, __builtin_return_address(0), caller_frame(), caller_frame2());
    return r;
}

cerr_t cudaMalloc(void **p, size_t n)
{
    cerr_t r = real_malloc(p, n);
    __atomic_add_fetch(&g_n_malloc, 1, __ATOMIC_RELAXED);
    if (r == 0)
        note_alloc(*p, n, __builtin_return_address(0), caller_frame(), caller_frame2());
    return r;
}

cerr_t cudaFreeAsync_ptsz(void *p, cstream_t s)
{
    __atomic_add_fetch(&g_n_free_async_ptsz, 1, __ATOMIC_RELAXED);
    note_free(p);
    return real_free_async_ptsz(p, s);
}

cerr_t cudaFreeAsync(void *p, cstream_t s)
{
    __atomic_add_fetch(&g_n_free_async, 1, __ATOMIC_RELAXED);
    note_free(p);
    return real_free_async(p, s);
}

cerr_t cudaFree(void *p)
{
    __atomic_add_fetch(&g_n_free, 1, __ATOMIC_RELAXED);
    note_free(p);
    return real_free(p);
}

static void copy_file(const char *from, const char *to)
{
    FILE *i = fopen(from, "r"), *o = fopen(to, "w");
    if (!i || !o) {
        if (i) fclose(i);
        if (o) fclose(o);
        return;
    }
    char buf[8192];
    size_t k;
    while ((k = fread(buf, 1, sizeof buf, i)) > 0)
        fwrite(buf, 1, k, o);
    fclose(i);
    fclose(o);
}

static void dump(void)
{
    if (!g_on)
        return;
    const char *dir = getenv("SNNI_CUDAREC_OUT");
    if (!dir || !*dir)
        dir = "/results";

    char path[512];
    snprintf(path, sizeof path, "%s/device_table.txt", dir);
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "SNNI_CUDAREC|dump_failed=cannot open %s\n", path);
        return;
    }

    /* Per-hook call counts: a hook that never fired is otherwise invisible in the table itself. */
    fprintf(f, "# hooks malloc_async_ptsz=%llu malloc_async=%llu malloc=%llu "
               "free_async_ptsz=%llu free_async=%llu free=%llu\n",
            (unsigned long long)g_n_malloc_async_ptsz, (unsigned long long)g_n_malloc_async,
            (unsigned long long)g_n_malloc, (unsigned long long)g_n_free_async_ptsz,
            (unsigned long long)g_n_free_async, (unsigned long long)g_n_free);
    fprintf(f, "# peak_device_bytes=%llu live_at_exit=%llu sites=%u drops=%llu "
               "untracked_frees=%llu stale_reuse=%llu\n",
            (unsigned long long)g_peak, (unsigned long long)g_live, g_nsites,
            (unsigned long long)g_drops, (unsigned long long)g_untracked_frees,
            (unsigned long long)g_stale_reuse);
    fprintf(f, "# source=cuda_alloc units=allocated_device_bytes "
               "note=pool_reserved_is_larger_see_MANIFEST\n");
    fprintf(f, "# depth=%d\n", g_depth);
    fprintf(f, "# resident_bytes allocated_bytes count site caller caller2\n");

    for (uint32_t i = 0; i < g_nsites; i++) {
        if (!g_peak_bytes_by_site[i])
            continue;
        fprintf(f, "%llu %llu %llu 0x%llx 0x%llx 0x%llx\n",
                (unsigned long long)g_peak_bytes_by_site[i],
                (unsigned long long)g_peak_bytes_by_site[i],
                (unsigned long long)g_peak_count_by_site[i],
                (unsigned long long)(uintptr_t)g_sites[i].addr,
                (unsigned long long)(uintptr_t)g_sites[i].caller,
                (unsigned long long)(uintptr_t)g_sites[i].caller2);
    }
    fclose(f);

    /* The reserved-high snapshot, when opted in: the same per-site live table, taken at the instant
     * device memory in use was highest, so it decomposes the moment that sets the reported peak.
     * The gap between peak_reserved_bytes and live_at_reserved_peak is pool retention plus
     * fragmentation, which carries no call site (see MANIFEST). Written to its own file so the
     * default device_table.txt and everything downstream of it are untouched. */
    if (g_rpeak_on) {
        snprintf(path, sizeof path, "%s/device_table_reservedpeak.txt", dir);
        FILE *g = fopen(path, "w");
        if (g) {
            fprintf(g, "# live per-site composition at the reserved high-water (device memory in use,\n");
            fprintf(g, "# total-free, at its maximum), i.e. the instant that sets the reported peak.\n");
            fprintf(g, "# peak_reserved_bytes=%llu live_at_reserved_peak=%llu sites=%u every=%llu\n",
                    (unsigned long long)g_peak_reserved, (unsigned long long)g_live_at_rpeak,
                    g_nsites, (unsigned long long)g_rpeak_every);
            fprintf(g, "# source=cuda_alloc units=allocated_device_bytes note=live_at_reserved_high\n");
            fprintf(g, "# depth=%d\n", g_depth);
            fprintf(g, "# resident_bytes allocated_bytes count site caller caller2\n");
            for (uint32_t i = 0; i < g_nsites; i++) {
                if (!g_rpeak_bytes_by_site[i])
                    continue;
                fprintf(g, "%llu %llu %llu 0x%llx 0x%llx 0x%llx\n",
                        (unsigned long long)g_rpeak_bytes_by_site[i],
                        (unsigned long long)g_rpeak_bytes_by_site[i],
                        (unsigned long long)g_rpeak_count_by_site[i],
                        (unsigned long long)(uintptr_t)g_sites[i].addr,
                        (unsigned long long)(uintptr_t)g_sites[i].caller,
                        (unsigned long long)(uintptr_t)g_sites[i].caller2);
            }
            fclose(g);
            fprintf(stderr, "SNNI_CUDAREC|peak_reserved_bytes=%llu|live_at_reserved_peak=%llu|wrote=%s\n",
                    (unsigned long long)g_peak_reserved, (unsigned long long)g_live_at_rpeak, dir);
        }
    }

    /* Maps snapshot travels with the addresses, meaningless once the process exits. Mappings are
     * fixed before the first CUDA call, so an exit-time snapshot matches the peak's layout. */
    snprintf(path, sizeof path, "%s/device.maps", dir);
    copy_file("/proc/self/maps", path);

    fprintf(stderr, "SNNI_CUDAREC|peak_device_bytes=%llu|sites=%u|drops=%llu|wrote=%s\n",
            (unsigned long long)g_peak, g_nsites, (unsigned long long)g_drops, dir);
}
