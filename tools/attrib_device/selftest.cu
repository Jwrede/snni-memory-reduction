/* Selftest for cudarec.so; expected values are fixed in advance.
 *
 * Built with --default-stream per-thread (redirects the runtime API to the _ptsz symbols, as in MOAI-GPU).
 *
 * Phase 1: A 64 MiB, B 128 MiB, C 256 MiB from three call sites, all live:
 *          peak = 469762048 B. Then free B, allocate D 32 MiB: peak unchanged.
 * Phase 2 (2026-08-08, pointer-table fix): churn of small buffers, half freed and replaced.
 *          Invariant: untracked_frees = 0, stale_reuse = 0, no site negative.
 * Phase 3 (2026-08-09, thread-safety fix): the same churn from many OpenMP threads.
 *          Invariant: live_at_exit = 0 after every pointer is freed.
 */
#include <cstdio>
#include <cuda_runtime.h>
#include <omp.h>

#define CK(x)                                                                    \
    do {                                                                         \
        cudaError_t _e = (x);                                                    \
        if (_e != cudaSuccess) {                                                 \
            printf("SELFTEST_FAIL cuda error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
            return 2;                                                            \
        }                                                                        \
    } while (0)

/* PHASE 4 tests the CALLER capture, and it is built so that depth 1 CANNOT pass it.
 *
 * All three of these allocate through ONE helper, exactly as the measured systems allocate through
 * `phantom::util::make_cuda_auto_ptr`. At SNNI_CUDAREC_DEPTH=1 the recorder keys on the helper and
 * the three collapse into a single site, which is the defect: 49.3% of MOAI-GPU's device peak sits
 * under such a helper and cannot be attacked. At depth 2 the key is the pair, so they separate.
 *
 * noinline, because the point is a real stack frame: if the compiler inlined the helper, addr2line
 * -i would already resolve it and there would be nothing to test. */
__attribute__((noinline)) static void *helper_alloc(size_t n, cudaStream_t s)
{
    void *p = nullptr;
    if (cudaMallocAsync(&p, n, s) != cudaSuccess)
        return nullptr;
    return p;
}

__attribute__((noinline)) static void *owner_a(cudaStream_t s) { return helper_alloc(1 << 20, s); }
__attribute__((noinline)) static void *owner_b(cudaStream_t s) { return helper_alloc(2 << 20, s); }
__attribute__((noinline)) static void *owner_c(cudaStream_t s) { return helper_alloc(3 << 20, s); }

int main()
{
    const size_t MiB = 1024ull * 1024ull;
    void *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    /* THE HELPER PHASE COMES FIRST AND STAYS LIVE THROUGH THE PEAK, because the recorder's table
     * is a snapshot of the PEAK INSTANT, not a summary of the run. A first version allocated and
     * freed these at the END, after the peak, so their rows were all zero and the check could not
     * see them at all -- the test was asking the table a question it does not answer. */
    void *ha = owner_a(s), *hb = owner_b(s), *hc = owner_c(s);
    CK(cudaStreamSynchronize(s));
    if (!ha || !hb || !hc) {
        printf("SELFTEST_FAIL helper allocation failed\n");
        return 2;
    }

    CK(cudaMallocAsync(&a, 64 * MiB, s));
    CK(cudaMallocAsync(&b, 128 * MiB, s));
    CK(cudaMallocAsync(&c, 256 * MiB, s));
    CK(cudaStreamSynchronize(s));

    /* 448 MiB from a+b+c, plus the 1+2+3 MiB the three helper owners hold across the whole run. */
    printf("SELFTEST_EXPECT peak_device_bytes=%llu sites_at_least=3\n",
           (unsigned long long)(454 * MiB));

    CK(cudaFreeAsync(b, s));
    CK(cudaMallocAsync(&d, 32 * MiB, s));
    CK(cudaStreamSynchronize(s));

    CK(cudaFreeAsync(a, s));
    CK(cudaFreeAsync(c, s));
    CK(cudaFreeAsync(d, s));
    CK(cudaStreamSynchronize(s));

    /* phase 2: churn. 4 KiB each, so 40,000 of them is 160 MiB and stays BELOW the 448 MiB peak
     * asserted above -- this phase must not move it, and a first draft at 64 KiB would have, which
     * is why the size is stated with its arithmetic. Freeing every second buffer and allocating a
     * replacement is what produces reopened slots, reused addresses and long probe chains. */
    const int N = 40000;
    static void *p[N];
    for (int i = 0; i < N; i++)
        CK(cudaMallocAsync(&p[i], 4 * 1024, s));
    CK(cudaStreamSynchronize(s));
    for (int round = 0; round < 3; round++) {
        for (int i = round % 2; i < N; i += 2) {
            CK(cudaFreeAsync(p[i], s));
            CK(cudaMallocAsync(&p[i], 4 * 1024, s));
        }
        CK(cudaStreamSynchronize(s));
    }
    for (int i = 0; i < N; i++)
        CK(cudaFreeAsync(p[i], s));
    CK(cudaStreamSynchronize(s));

    /* phase 3: the same churn, concurrently. Each thread gets its own stream and its own slice,
     * so the program has no race of its own and anything that drifts is the recorder's. */
    const int T = 8, PER = 2000;
    static void *q[T][PER];
    int fail = 0;
#pragma omp parallel num_threads(T) reduction(| : fail)
    {
        int t = omp_get_thread_num();
        cudaStream_t ts;
        if (cudaStreamCreateWithFlags(&ts, cudaStreamNonBlocking) != cudaSuccess)
            fail = 1;
        for (int r = 0; r < 4 && !fail; r++) {
            for (int i = 0; i < PER; i++)
                if (cudaMallocAsync(&q[t][i], 4 * 1024, ts) != cudaSuccess)
                    fail = 1;
            if (cudaStreamSynchronize(ts) != cudaSuccess)
                fail = 1;
            for (int i = 0; i < PER; i++)
                if (cudaFreeAsync(q[t][i], ts) != cudaSuccess)
                    fail = 1;
            if (cudaStreamSynchronize(ts) != cudaSuccess)
                fail = 1;
        }
        cudaStreamDestroy(ts);
    }
    if (fail) {
        printf("SELFTEST_FAIL cuda error in the parallel churn\n");
        return 2;
    }

    CK(cudaFreeAsync(ha, s));
    CK(cudaFreeAsync(hb, s));
    CK(cudaFreeAsync(hc, s));
    CK(cudaStreamSynchronize(s));

    CK(cudaStreamDestroy(s));
    printf("SELFTEST_CHURN_DONE allocs=%d rounds=3 par_threads=%d par_each=%d par_rounds=4 "
           "helper_owners=3\n", N, T, PER);
    printf("SELFTEST_DONE\n");
    return 0;
}
