// pmsample: resident bytes per recorded allocation, aggregated by call site.
// Reads pmrec.so's shared table; present pages from /proc/<pid>/pagemap. 4 GiB scans in ~31 ms.
//
// Usage: pmsample <pid> <table> [maps-out]
//   stdout: one line per call site   resident_bytes allocated_bytes count 0xsite 0xcaller
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define SLOTS (1u << 17)
/* Two table layouts read deliberately: C3 carries the frame chain, C2 (retired) carries only a
 * single caller. A run that started under a C2 recorder must still be derivable when it ends. */
#define MAGIC_C3 0x504d52454333ULL
#define MAGIC_C2 0x504d52454332ULL
#define PM_FRAMES 12

struct slot_c2 { uint64_t ptr, size, site, caller; };
struct slot_c3 { uint64_t ptr, size, site, caller, frames[PM_FRAMES]; };
struct table { uint64_t magic, inserts, drops, slots; unsigned char s[]; };

struct agg { uint64_t site, caller, res, alloc, n, frames[PM_FRAMES]; };

static int cmp_rng(const void *a, const void *b)
{
    uint64_t x = ((const struct { uint64_t start, end; } *)a)->start;
    uint64_t y = ((const struct { uint64_t start, end; } *)b)->start;
    return x < y ? -1 : x > y;
}

static int cmp(const void *a, const void *b)
{
    uint64_t x = ((const struct agg *)a)->res, y = ((const struct agg *)b)->res;
    return x < y ? 1 : x > y ? -1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: pmsample <pid> <table> [maps-out]\n"); return 2; }
    int pid = atoi(argv[1]);

    int tf = open(argv[2], O_RDONLY);
    if (tf < 0) { perror("table"); return 1; }
    struct stat st; fstat(tf, &st);
    struct table *t = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, tf, 0);
    if (t == MAP_FAILED || (t->magic != MAGIC_C3 && t->magic != MAGIC_C2)) {
        fprintf(stderr, "table not ready\n"); return 1;
    }
    const int c3 = (t->magic == MAGIC_C3);
    const size_t slotsz = c3 ? sizeof(struct slot_c3) : sizeof(struct slot_c2);

    char pmpath[64]; snprintf(pmpath, sizeof pmpath, "/proc/%d/pagemap", pid);
    int pm = open(pmpath, O_RDONLY);
    if (pm < 0) { perror("pagemap"); return 1; }

    /* Call site is a runtime address, meaningless post-exit under ASLR without the maps snapshot
       captured alongside it. */
    if (argc > 3) {
        char mp[64]; snprintf(mp, sizeof mp, "/proc/%d/maps", pid);
        FILE *in = fopen(mp, "r"), *out = fopen(argv[3], "w");
        if (in && out) { char b[4096]; size_t n; while ((n = fread(b, 1, sizeof b, in)) > 0) fwrite(b, 1, n, out); }
        if (in) fclose(in); if (out) fclose(out);
    }

    static struct agg agg[SLOTS]; size_t na = 0;
    uint64_t *buf = malloc(65536 * sizeof(uint64_t));
    uint64_t live = 0, live_alloc = 0;

    /* Boundary accounting: an allocation not aligned to a page shares its first/last page with a
       neighbour, double-counting that page if the neighbour is also tracked. Not designed away
       (would mean replacing the allocator, changing the layout being measured); measured and
       published instead via shared_boundary_pages below. */
    static struct rng { uint64_t start, end; } rng[SLOTS]; size_t nrange = 0;

    for (uint32_t i = 0; i < SLOTS; i++) {
        const struct slot_c2 *sl = (const struct slot_c2 *)(t->s + (size_t)i * slotsz);
        const uint64_t *sfr = c3 ? ((const struct slot_c3 *)sl)->frames : NULL;
        uint64_t p = sl->ptr, sz = sl->size, site = sl->site, caller = sl->caller;
        if (!p || !sz) continue;
        live++; live_alloc += sz;

        uint64_t first = p / 4096, npg = (p + sz + 4095) / 4096 - first, present = 0, done = 0;
        while (done < npg) {
            size_t want = npg - done; if (want > 65536) want = 65536;
            ssize_t got = pread(pm, buf, want * 8, (off_t)(first + done) * 8);
            if (got <= 0) break;
            size_t k = got / 8;
            for (size_t j = 0; j < k; j++) if (buf[j] & (1ULL << 63)) present++;
            done += k;
        }
        uint64_t res = present * 4096;
        if (nrange < SLOTS) { rng[nrange].start = first; rng[nrange].end = first + npg; nrange++; }

        size_t a = 0;
        /* Keyed on the pair AND, for a C3 table, the whole recorded chain. The pair alone assumed
           one (site, caller) is one call path; for the SEAL pool recorder that pair is the same
           (get_for_byte_count, allocate<uint64>) for every ciphertext, so every ciphertext of the
           process collapsed into ONE row named after whichever slot came first (MOAI-CPU m8:
           4554 items, 104.8 GB, labelled by a 2 MB matmul output). resolve_resident.py merges rows
           by resolved label, so finer rows cost nothing downstream. */
        for (; a < na; a++) {
            if (agg[a].site != site || agg[a].caller != caller) continue;
            if (!sfr) break;
            int same = 1;
            for (int f = 0; f < PM_FRAMES; f++) if (agg[a].frames[f] != sfr[f]) { same = 0; break; }
            if (same) break;
        }
        if (a == na) { if (na >= SLOTS) { fprintf(stderr, "agg full\n"); return 1; }
                       agg[na].site = site; agg[na].caller = caller;
                       agg[na].res = agg[na].alloc = agg[na].n = 0;
                       for (int f = 0; f < PM_FRAMES; f++)
                           agg[na].frames[f] = sfr ? sfr[f] : 0;
                       na++; }
        agg[a].res += res; agg[a].alloc += sz; agg[a].n++;
    }

    /* Sort by start and sweep once: O(n log n) rather than pairwise O(n^2), which at ~3500 live
       ranges would cost more than the scan it reports on. */
    qsort(rng, nrange, sizeof rng[0], cmp_rng);
    uint64_t shared = 0, counted = 0, reach = 0;
    for (size_t i = 0; i < nrange; i++) {
        counted += rng[i].end - rng[i].start;
        if (rng[i].start < reach) {
            uint64_t hi = rng[i].end < reach ? rng[i].end : reach;
            shared += hi - rng[i].start;
        }
        if (rng[i].end > reach) reach = rng[i].end;
    }

    /* Full-address-space scan (SNNI_PM_FULLSCAN=1): by default only recorder-known ranges are
       scanned (tens of ms); everything else is a subtracted "residual" (on the LLAMA dealer,
       ~500 MB at every step and most of the peak at the last). This scans every mapping too, so
       the residual gets a location (mapping, size, anon/file-backed) though still not a who --
       pagemap answers "what is here", never "who put it there".
       PROT_NONE mappings are skipped by default: the scan costs address space, not RSS (measured
       36x slowdown, 17ms->620ms, on a 512 GiB PROT_NONE reservation with nothing in it). The skip
       is verified via SNNI_PM_FULLSCAN=all (0 resident bytes there, confirming it), and the
       skipped total is always printed rather than silently dropped. */
    /* `fs && *fs`, not `fs`: an unset env var passed through as an empty string must not switch
       this on (this campaign has paid for that distinction once already). */
    const char *fs = getenv("SNNI_PM_FULLSCAN");
    if (fs && *fs) {
        int scan_noperm = strcmp(fs, "all") == 0;
        uint64_t skipped_bytes = 0, skipped_n = 0;
        char mpath[64]; snprintf(mpath, sizeof mpath, "/proc/%d/maps", pid);
        FILE *mf = fopen(mpath, "r");
        if (mf) {
            char line[512];
            printf("# fullscan: mapping_resident_bytes unaccounted_bytes perms path\n");
            while (fgets(line, sizeof line, mf)) {
                unsigned long lo, hi; char perms[8] = "", path[256] = "";
                if (sscanf(line, "%lx-%lx %7s %*s %*s %*s %255[^\n]", &lo, &hi, perms, path) < 3)
                    continue;
                if (!scan_noperm && perms[0] == '-' && perms[1] == '-') {
                    skipped_bytes += hi - lo; skipped_n++;
                    continue;
                }
                uint64_t f = lo / 4096, n = (hi + 4095) / 4096 - f, pres = 0, d = 0;
                while (d < n) {
                    size_t want = n - d; if (want > 65536) want = 65536;
                    ssize_t got = pread(pm, buf, want * 8, (off_t)(f + d) * 8);
                    if (got <= 0) break;
                    for (size_t j = 0; j < (size_t)got / 8; j++)
                        if (buf[j] & (1ULL << 63)) pres++;
                    d += got / 8;
                }
                if (!pres) continue;
                /* how many present pages here are already charged to a recorded allocation */
                uint64_t known = 0;
                for (size_t i = 0; i < nrange; i++) {
                    uint64_t a = rng[i].start > f ? rng[i].start : f;
                    uint64_t b = rng[i].end < f + n ? rng[i].end : f + n;
                    if (b > a) known += b - a;
                }
                if (known > pres) known = pres;   /* recorded ranges include untouched pages */
                printf("# map %llu %llu %s %s\n", (unsigned long long)pres * 4096,
                       (unsigned long long)(pres - known) * 4096, perms, path);
            }
            fclose(mf);
        }
        /* Always printed, including as zero: a consumer must be able to tell "nothing was skipped"
           from "this build did not report skipping". */
        printf("# fullscan_skipped_bytes=%llu mappings=%llu mode=%s\n",
               (unsigned long long)skipped_bytes, (unsigned long long)skipped_n,
               scan_noperm ? "all" : "readable-only");
    }

    qsort(agg, na, sizeof agg[0], cmp);
    printf("# shared_boundary_pages=%llu of %llu counted (%.3f%%)\n",
           (unsigned long long)shared, (unsigned long long)counted,
           counted ? 100.0 * (double)shared / (double)counted : 0.0);
    printf("# live_allocations=%llu allocated_bytes=%llu inserts=%llu drops=%llu\n",
           (unsigned long long)live, (unsigned long long)live_alloc,
           (unsigned long long)t->inserts, (unsigned long long)t->drops);
    /* The frame chain is APPENDED, so a reader that knows only the five original columns keeps
       working on a new table and a reader that knows the chain keeps working on an old one. */
    printf("# resident_bytes allocated_bytes count site caller [frames...]\n");
    for (size_t i = 0; i < na; i++) {
        printf("%llu %llu %llu 0x%llx 0x%llx", (unsigned long long)agg[i].res,
               (unsigned long long)agg[i].alloc, (unsigned long long)agg[i].n,
               (unsigned long long)agg[i].site, (unsigned long long)agg[i].caller);
        for (int f = 0; f < PM_FRAMES && agg[i].frames[f]; f++)
            printf(" 0x%llx", (unsigned long long)agg[i].frames[f]);
        printf("\n");
    }
    return 0;
}
