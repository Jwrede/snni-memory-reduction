/* Selftest for pmrec; expected values fixed in advance. All sizes above SNNI_PM_MIN.
 * Phase 1 (accounting): three allocations of known size from three call sites, all live.
 * Phase 2 (caller frame): three owners allocate through one noinline helper: one row at depth 1,
 * three rows at depth 2.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MiB (1024UL * 1024UL)

/* noinline: the point is a real stack frame; an inlined helper would pass for the wrong reason. */
__attribute__((noinline)) static void *helper_alloc(size_t n)
{
    void *p = malloc(n);
    if (p)
        memset(p, 1, 4096); /* touch one page, so the sampler sees it resident */
    return p;
}

__attribute__((noinline)) static void *owner_a(void) { return helper_alloc(8 * MiB); }
__attribute__((noinline)) static void *owner_b(void) { return helper_alloc(16 * MiB); }
__attribute__((noinline)) static void *owner_c(void) { return helper_alloc(32 * MiB); }

int main(void)
{
    /* phase 2 first and held across phase 1, because the sampler reads the table at an instant:
       anything freed before that instant is simply not there to check. */
    void *ha = owner_a(), *hb = owner_b(), *hc = owner_c();
    if (!ha || !hb || !hc) {
        printf("SELFTEST_FAIL helper allocation returned NULL\n");
        return 2;
    }

    void *a = malloc(64 * MiB);
    void *b = malloc(128 * MiB);
    void *c = malloc(256 * MiB);
    if (!a || !b || !c) {
        printf("SELFTEST_FAIL direct allocation returned NULL\n");
        return 2;
    }
    memset(a, 1, 4096);
    memset(b, 1, 4096);
    memset(c, 1, 4096);

    printf("SELFTEST_READY live_alloc_bytes=%llu helper_owners=3 direct_sites=3\n",
           (unsigned long long)((8 + 16 + 32 + 64 + 128 + 256) * MiB));
    fflush(stdout);

    /* Hold everything while the sampler runs, then release so a leak check can be exact. */
    if (getenv("SNNI_SELFTEST_HOLD_SECONDS")) {
        int secs = atoi(getenv("SNNI_SELFTEST_HOLD_SECONDS"));
        for (int i = 0; i < secs; i++) {
            struct timespec ts = {1, 0};
            nanosleep(&ts, NULL);
        }
    }

    free(a); free(b); free(c);
    free(ha); free(hb); free(hc);
    printf("SELFTEST_DONE\n");
    return 0;
}
