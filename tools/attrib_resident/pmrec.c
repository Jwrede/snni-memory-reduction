// pmrec: records address range and call site of every allocation >= SNNI_PM_MIN (default 1 MiB),
// so pmsample can read resident bytes at the peak.
// Interceptor: no allocation, no signal handlers, shared-mmap fixed-size table, no in-process
// symbolisation; second frame opt-in via SNNI_PM_DEPTH=2 (caller_frame()).
//
// Usage: LD_PRELOAD=pmrec.so SNNI_PM_TABLE=/path/table SNNI_PM_MIN=1048576 <program>
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>

#define SLOTS (1u << 17)          /* 131072 slots, ample for large-allocation live sets */
#define MAGIC 0x504d52454333ULL  /* ...C3: the slot gained the frame CHAIN; a C2 sampler must not
                                    read this. A C2 table is still readable BY the current sampler,
                                    which is deliberate: a run that started under the old recorder
                                    must still be derivable when it ends. */

/* Frames above the allocation kept per slot. SNNI_PM_DEPTH names one frame by INDEX, which is the
 * wrong handle: how far above the allocator program code sits varies by call chain, so a fixed
 * index reaches program code in one binary and stops inside the library in another (measured on
 * MOAI-CPU across a one-lever image change). The whole chain is recorded instead, and the choice
 * is made at RESOLVE time via addr2line's source path, which is stable across images. Twelve:
 * margin above what reached program code in the worst measured case. */
#define PM_FRAMES 12

struct slot {
    volatile uintptr_t ptr;       /* 0 = empty */
    uint64_t size;
    uint64_t site;                /* return address of the caller */
    uint64_t caller;              /* the frame at SNNI_PM_DEPTH, kept so the retired fixed-index
                                     rule still reads exactly as it did */
    uint64_t frames[PM_FRAMES];   /* the chain ABOVE site, innermost first, 0-padded */
};

struct table {
    uint64_t magic;
    volatile uint64_t inserts, drops;
    uint64_t slots;
    struct slot s[];
};

static struct table *tab;
/* 64 KiB floor, measured not chosen: the floor decides what the ranking can see. Criterion: the
 * ranking must be stable across the last halving. On the LLAMA dealer, the top-two ranking
 * inverted between 256 KiB and 64 KiB and then stopped moving; at 1 MiB it was invisibly wrong
 * (a 2715-allocation, 35%-touched object was entirely below the floor). */
static size_t min_size = 1u << 16;
/* SNNI_PM_DEPTH: 1 = allocating function only, N = plus the N-1'th frame above it. Ceiling
 * bounds unwind cost, not stack depth. It used to clamp silently (fixed 2026-08-22, cost a
 * MOAI-CPU depth-7 request four documents reasoned about without any run ever producing it): the
 * effective depth is now always announced on stderr, and the ceiling is raised only via the
 * separate SNNI_PM_DEPTH_MAX, e.g. `SNNI_PM_DEPTH_MAX=12 SNNI_PM_DEPTH=10`, so a queued job's
 * meaning can't silently change because the tool was rebuilt. */
#define PM_HARD_MAX_DEPTH 12
#define PM_DEFAULT_MAX_DEPTH 6
static int pm_depth = 1;
static int pm_depth_requested = 1;

/* Filled by caller_frame() and read by record() rather than passed as an argument: every
 * interceptor evaluates caller_frame() in the same expression, and `busy` rules out reentrancy.
 * Thread-local since interceptors aren't serialised. */
static __thread uint64_t tls_frames[PM_FRAMES];

/* Written with write(2), not fprintf: this runs inside init(), reached during glibc's own dlsym
 * bootstrap, and stdio can allocate there, which this file forbids. */
static void pm_num(char *b, size_t *n, int v)
{
    if (v >= 10)
        b[(*n)++] = (char)('0' + v / 10);
    b[(*n)++] = (char)('0' + v % 10);
}

static void pm_str(char *b, size_t *n, const char *s)
{
    while (*s)
        b[(*n)++] = *s++;
}

static void pm_say_depth(void)
{
    char b[96];
    size_t n = 0;
    pm_str(b, &n, "SNNI_PMREC|depth_requested=");
    pm_num(b, &n, pm_depth_requested);
    pm_str(b, &n, "|depth_effective=");
    pm_num(b, &n, pm_depth);
    pm_str(b, &n, pm_depth_requested > pm_depth ? "|clamped=1\n" : "|clamped=0\n");
    ssize_t w = write(2, b, n);
    (void)w;
}

/* Bootstrap arena: init()'s dlsym() call triggers glibc's own calloc, which without this would
 * re-enter init() and recurse unboundedly (measured: SIGSEGV on every process on Ubuntu
 * 20.04/glibc 2.31, absent on the glibc-2.35 image this was first tested against). Fixed-size on
 * purpose -- an earlier shim's growing arena was one of the things that made it unusable. */
#define BOOT_BYTES 65536
static char boot_arena[BOOT_BYTES];
static size_t boot_used;
static int in_init;

static int from_boot(const void *p)
{
    return (const char *)p >= boot_arena && (const char *)p < boot_arena + BOOT_BYTES;
}

static void *boot_alloc(size_t sz)
{
    sz = (sz + 15u) & ~(size_t)15;             /* keep max_align_t alignment */
    if (boot_used + sz > BOOT_BYTES) return NULL;
    void *p = boot_arena + boot_used;
    boot_used += sz;
    return p;
}
static void *(*real_malloc)(size_t);
static void *(*real_calloc)(size_t, size_t);
static void *(*real_realloc)(void *, size_t);
static void  (*real_free)(void *);
static int  (*real_pma)(void **, size_t, size_t);
/* aligned_alloc and memalign share one underlying entry point in glibc. Resolved
   separately from malloc because a program can call them without ever calling malloc. */
static void *(*real_memalign)(size_t, size_t);
/* initial-exec TLS model, not the default: a plain __thread here gets the dynamic model, whose
 * block is malloc'd on first access -- from inside our own malloc wrapper. Measured: worked on a
 * small C++ binary and Ubuntu 20.04 torch, then killed `import torch` on glibc 2.39. initial-exec
 * allocates nothing, at the cost of one static-TLS slot. */
static __thread int busy __attribute__((tls_model("initial-exec")));

static inline uint32_t hsh(uintptr_t p) { p >>= 4; p *= 0x9E3779B97F4A7C15ULL; return (uint32_t)(p >> 32) & (SLOTS - 1); }

static void init(void)
{
    if (in_init) return;                       /* re-entered through dlsym; the outer call wins */
    in_init = 1;
    real_malloc  = dlsym(RTLD_NEXT, "malloc");
    real_calloc  = dlsym(RTLD_NEXT, "calloc");
    real_realloc = dlsym(RTLD_NEXT, "realloc");
    real_free    = dlsym(RTLD_NEXT, "free");
    real_pma     = dlsym(RTLD_NEXT, "posix_memalign");
    real_memalign = dlsym(RTLD_NEXT, "memalign");
    if (!real_memalign) real_memalign = dlsym(RTLD_NEXT, "aligned_alloc");
    /* Depth goes above 2 because on some systems the second frame is still inside the same
     * third-party library (measured: SHAFT-CPU's second frame was numpy calling numpy).
     * Depths above the ceiling clamp rather than fail; the clamp is announced (see the ceiling's
     * own comment for what a silent one cost). */
    int pm_max = PM_DEFAULT_MAX_DEPTH;
    const char *dm = getenv("SNNI_PM_DEPTH_MAX");
    if (dm) {
        long m = strtol(dm, NULL, 10);
        if (m < 1) m = 1;
        if (m > PM_HARD_MAX_DEPTH) m = PM_HARD_MAX_DEPTH;
        pm_max = (int)m;
    }
    const char *d = getenv("SNNI_PM_DEPTH");
    if (d) {
        long v = strtol(d, NULL, 10);
        if (v < 1) v = 1;
        pm_depth_requested = (int)(v > 99 ? 99 : v);
        if (v > pm_max) v = pm_max;
        pm_depth = (int)v;
    }
    pm_say_depth();
    if (pm_depth > 1) {
        /* Warmed here, not in the hot path: backtrace() allocates on its first call (loading
         * libgcc's unwinder), and allocation is forbidden in the interceptor. */
        void *warm[PM_HARD_MAX_DEPTH + 2];
        (void)backtrace(warm, PM_HARD_MAX_DEPTH + 2);
    }

    const char *m = getenv("SNNI_PM_MIN");
    if (m) min_size = strtoul(m, NULL, 10);
    const char *path = getenv("SNNI_PM_TABLE");
    if (!path) { in_init = 0; return; }
    /* `%d` expands to this process's pid: without it, multi-party systems would silently
     * truncate each other's table. The poller substitutes the same way. */
    char pathbuf[4096];
    const char *pct = strstr(path, "%d");
    if (pct) {
        size_t head = (size_t)(pct - path);
        if (head < sizeof(pathbuf) - 32) {
            memcpy(pathbuf, path, head);
            int n = snprintf(pathbuf + head, sizeof(pathbuf) - head, "%d%s",
                             (int)getpid(), pct + 2);
            if (n > 0 && (size_t)n < sizeof(pathbuf) - head) path = pathbuf;
        }
    }
    size_t bytes = sizeof(struct table) + (size_t)SLOTS * sizeof(struct slot);
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { in_init = 0; return; }
    if (ftruncate(fd, bytes) != 0) { close(fd); in_init = 0; return; }
    void *m2 = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m2 == MAP_FAILED) { in_init = 0; return; }
    tab = m2;
    tab->slots = SLOTS;
    tab->magic = MAGIC;          /* written last: the sampler treats it as the ready flag */
    in_init = 0;
}

/* Second frame, off by default: `site` (the allocator's caller) is often third-party code with
 * no debug info (e.g. SHAFT-CPU's `libc10.so+0x4a675`); the frame above it is usually in the
 * measured program. backtrace() is used instead of __builtin_return_address(1), which GCC
 * documents as able to crash without frame pointers -- exactly the case here. Stays within this
 * file's no-allocation rule: warmed once in init(), and every caller already holds `busy`. */
__attribute__((noinline)) static void *caller_frame(void)
{
    for (int f = 0; f < PM_FRAMES; f++)
        tls_frames[f] = 0;
    if (pm_depth < 2)
        return NULL;
    void *bt[PM_HARD_MAX_DEPTH + PM_FRAMES + 4];
    int want = pm_depth + 1;              /* depth 2 -> bt[3], depth 3 -> bt[4], and so on */
    int deep = 3 + PM_FRAMES;             /* the whole chain above `site`, for the resolver */
    int n = backtrace(bt, (want + 1 > deep ? want + 1 : deep));
    /* noinline so the frame index can't depend on optimiser inlining (a prior inlined version
       shifted every index by one, collapsing three owners into a single row). bt[0]=this fn,
       bt[1]=allocator, bt[2]=site, bt[3]=wanted frame at depth 2; depth N wants index N+1. The
       full chain from bt[3] is recorded too, so the resolver can choose a frame by identity. */
    for (int f = 0; f < PM_FRAMES && 3 + f < n; f++)
        tls_frames[f] = (uint64_t)(uintptr_t)bt[3 + f];
    return n > want ? bt[want] : NULL;
}

static void record(void *p, size_t sz, void *site, void *caller)
{
    if (!tab || !p || sz < min_size) return;
    uint32_t i = hsh((uintptr_t)p);
    for (uint32_t n = 0; n < 64; n++, i = (i + 1) & (SLOTS - 1)) {
        uintptr_t exp = 0;
        if (__atomic_compare_exchange_n(&tab->s[i].ptr, &exp, (uintptr_t)p, 0,
                                        __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            tab->s[i].size = sz;
            tab->s[i].site = (uint64_t)(uintptr_t)site;
            tab->s[i].caller = (uint64_t)(uintptr_t)caller;
            for (int f = 0; f < PM_FRAMES; f++)
                tab->s[i].frames[f] = tls_frames[f];
            __atomic_fetch_add(&tab->inserts, 1, __ATOMIC_RELAXED);
            return;
        }
    }
    __atomic_fetch_add(&tab->drops, 1, __ATOMIC_RELAXED);  /* never silently lose count */
}

static void forget(void *p)
{
    if (!tab || !p) return;
    uint32_t i = hsh((uintptr_t)p);
    for (uint32_t n = 0; n < 64; n++, i = (i + 1) & (SLOTS - 1)) {
        if (__atomic_load_n(&tab->s[i].ptr, __ATOMIC_ACQUIRE) == (uintptr_t)p) {
            tab->s[i].size = 0;
            __atomic_store_n(&tab->s[i].ptr, 0, __ATOMIC_RELEASE);
            return;
        }
    }
}

void *malloc(size_t sz)
{
    if (!real_malloc) init();
    if (!real_malloc) return boot_alloc(sz);   /* still inside init: dlsym is asking */
    void *p = real_malloc(sz);
    if (!busy) { busy = 1; record(p, sz, __builtin_return_address(0), caller_frame()); busy = 0; }
    return p;
}

void *calloc(size_t n, size_t sz)
{
    if (!real_calloc) init();
    if (!real_calloc) {
        void *b = boot_alloc(n * sz);
        if (b) memset(b, 0, n * sz);
        return b;
    }
    void *p = real_calloc(n, sz);
    if (!busy) { busy = 1; record(p, n * sz, __builtin_return_address(0), caller_frame()); busy = 0; }
    return p;
}

void *realloc(void *old, size_t sz)
{
    if (!real_realloc) init();
    if (from_boot(old)) {                      /* bootstrap block outgrown: copy it out */
        void *p = real_malloc ? real_malloc(sz) : boot_alloc(sz);
        if (p && old) memcpy(p, old, sz);
        return p;
    }
    if (!real_realloc) return boot_alloc(sz);
    void *p = real_realloc(old, sz);
    if (!busy) { busy = 1; if (old) forget(old); record(p, sz, __builtin_return_address(0), caller_frame()); busy = 0; }
    return p;
}

int posix_memalign(void **out, size_t al, size_t sz)
{
    if (!real_pma) init();
    if (!real_pma) { *out = boot_alloc(sz + al); return *out ? 0 : 12 /* ENOMEM */; }
    int r = real_pma(out, al, sz);
    if (!r && !busy) { busy = 1; record(*out, sz, __builtin_return_address(0), caller_frame()); busy = 0; }
    return r;
}

/* C++ goes through operator new; hooking malloc alone would collapse every allocation onto one
 * site (libstdc++'s new). These are the Itanium-ABI mangled names for new/new[]/delete/delete[]
 * and C++14's sized-delete forms. */
void *_Znwm(size_t sz)
{
    if (!real_malloc) init();
    if (!real_malloc) return boot_alloc(sz);
    void *p = real_malloc(sz);
    if (!p) abort();                      /* operator new must not return NULL; fail loudly */
    if (!busy) { busy = 1; record(p, sz, __builtin_return_address(0), caller_frame()); busy = 0; }
    return p;
}

void *_Znam(size_t sz)
{
    if (!real_malloc) init();
    if (!real_malloc) return boot_alloc(sz);
    void *p = real_malloc(sz);
    if (!p) abort();
    if (!busy) { busy = 1; record(p, sz, __builtin_return_address(0), caller_frame()); busy = 0; }
    return p;
}

/* Aligned allocation was a blind spot until 2026-07-30: aligned_alloc/memalign/valloc and the
 * C++17 over-aligned/nothrow new overloads were unwrapped. SEAL routes 64-byte-aligned blocks
 * through aligned_alloc; on BumbleBee this hid 7.93 GB with zero recorded allocations. Affects
 * BumbleBee, BOLT and MOAI (all use SEAL); check with
 * `nm -D --undefined-only <binary> | grep aligned_alloc` and re-measure if it hits. */
void *aligned_alloc(size_t align, size_t sz)
{
    if (!real_malloc) init();
    if (!real_memalign) return boot_alloc(sz);
    void *p = real_memalign(align, sz);
    if (p && !busy) { busy = 1; record(p, sz, __builtin_return_address(0), caller_frame()); busy = 0; }
    return p;
}

void *memalign(size_t align, size_t sz)
{
    if (!real_malloc) init();
    if (!real_memalign) return boot_alloc(sz);
    void *p = real_memalign(align, sz);
    if (p && !busy) { busy = 1; record(p, sz, __builtin_return_address(0), caller_frame()); busy = 0; }
    return p;
}

/* C++17 over-aligned new(size_t, align_val_t) and array/nothrow variants; libstdc++ routes these
 * to aligned_alloc on glibc, but a differently-linked binary may call them directly. */
void *_ZnwmSt11align_val_t(size_t sz, size_t align)      { return aligned_alloc(align, sz); }
void *_ZnamSt11align_val_t(size_t sz, size_t align)      { return aligned_alloc(align, sz); }
void *_ZnwmRKSt9nothrow_t(size_t sz, const void *nt)     { (void)nt; if (!real_malloc) init();
                                                           if (!real_malloc) return boot_alloc(sz);
                                                           void *p = real_malloc(sz);
                                                           if (p && !busy) { busy = 1; record(p, sz, __builtin_return_address(0), caller_frame()); busy = 0; }
                                                           return p; }
void *_ZnamRKSt9nothrow_t(size_t sz, const void *nt)     { return _ZnwmRKSt9nothrow_t(sz, nt); }

void _ZdlPv(void *p)  { if (from_boot(p)) return; if (!real_free) init(); if (!real_free) return;
                        if (!busy) { busy = 1; forget(p); busy = 0; } real_free(p); }
void _ZdaPv(void *p)  { if (from_boot(p)) return; if (!real_free) init(); if (!real_free) return;
                        if (!busy) { busy = 1; forget(p); busy = 0; } real_free(p); }
void _ZdlPvm(void *p, size_t sz) { (void)sz; _ZdlPv(p); }
void _ZdaPvm(void *p, size_t sz) { (void)sz; _ZdaPv(p); }

void free(void *p)
{
    if (from_boot(p)) return;                  /* never handed out by the real allocator */
    if (!real_free) init();
    if (!real_free) return;
    if (!busy) { busy = 1; forget(p); busy = 0; }
    real_free(p);
}
