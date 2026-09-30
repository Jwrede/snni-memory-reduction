#!/usr/bin/env python3
"""s7_weight_window: a layer's weights (679.9 MB, 32.5% of peak) resident only while the layer
computes; madvise(MADV_DONTNEED) after its forward, refilled before the next (deterministic fill,
snni_fill_deterministic). Per-tensor checksum recomputed on every refill, mismatch aborts (the
structural gate cannot see values). Paid (fill runs again).
"""

import os
import sys

H = "ext/sytorch/include/sytorch/layers/layers.h"
CU = "experiments/sigma/sigma.cu"

# ---- layers.h: two hooks around the one place every layer's execution passes ----
HOOK_OLD = '''        _forward(a);
        // printf("Layer=%s, doTruncationForward=%d\\n", this->name.data(), this->doTruncationForward);
'''
HOOK_NEW = '''        // s7_weight_window: the ONE choke point every layer's execution passes. `forward` is called
        // per node in topological order, so a hook here is per-layer without touching any model.
        // The hooks are null unless an experiment sets them, so every other program built from this
        // header is byte-identical in behaviour.
        if (snni_weights_in<T>) snni_weights_in<T>(this);
        _forward(a);
        if (snni_weights_out<T>) snni_weights_out<T>(this);
        // printf("Layer=%s, doTruncationForward=%d\\n", this->name.data(), this->doTruncationForward);
'''

# The declaration goes ABOVE the class (hooks take Layer<T> *). inline variable templates (C++17)
# give one definition across translation units without a .cpp.
DECL_OLD = '''template <typename T>
class Layer
{
'''
DECL_NEW = '''// s7_weight_window: per-layer hooks around `_forward`, null by default.
template <typename T>
class Layer;
template <typename T>
inline void (*snni_weights_in)(Layer<T> *) = nullptr;
template <typename T>
inline void (*snni_weights_out)(Layer<T> *) = nullptr;

template <typename T>
class Layer
{
'''

# ---- sigma.cu: record the fill, then drop and rewrite per layer ----
FILL_OLD = '''    u64 k = 0, wt = 0, welems = 0, nodes = 0;
'''
FILL_NEW = '''    u64 k = 0, wt = 0, welems = 0, nodes = 0;
    snni_ww_reset();
'''

# The per-tensor record is taken inside the existing walk, so the counters cannot drift from the
# fill they describe.
RECORD_OLD = '''            ++wt;
            welems += w.size;
            for (u64 i = 0; i < w.size; ++i) w.data[i] = (T)snni_fixed_value(k++, scale, 28000);
        }
        auto b = node->layer->getbias();
        if (b.data != nullptr)
            for (u64 i = 0; i < b.size; ++i) b.data[i] = (T)snni_fixed_value(k++, scale, 28000);
'''
RECORD_NEW = '''            ++wt;
            welems += w.size;
            u64 kw = k;
            for (u64 i = 0; i < w.size; ++i) w.data[i] = (T)snni_fixed_value(k++, scale, 28000);
            snni_ww_record((void *)node->layer, (void *)w.data, w.size, kw, true, scale);
            snni_ww_drop(snni_ww_regions.back());
        }
        auto b = node->layer->getbias();
        if (b.data != nullptr)
        {
            u64 kb = k;
            for (u64 i = 0; i < b.size; ++i) b.data[i] = (T)snni_fixed_value(k++, scale, 28000);
            snni_ww_record((void *)node->layer, (void *)b.data, b.size, kb, false, scale);
            snni_ww_drop(snni_ww_regions.back());
        }
'''

# The bookkeeping itself, placed above `snni_fill_deterministic` so the walk can call it.
CORE_OLD = '''template <typename T>
static void snni_fill_deterministic(SytorchModule<T> *net, Tensor<T> &input, u64 scale)
{
'''
CORE_NEW = '''// ---------------------------------------------------------------------------------------------
// s7_weight_window: a layer's weights are resident only while that layer is computing.
//
// The pages are ANONYMOUS heap, so `madvise(MADV_DONTNEED)` discards them at once and the next
// touch returns zeros. Zero is safe here because the fill is a pure function of a counter, so the
// bytes can be rewritten exactly. The checksum is what makes that a claim rather than a hope: this
// system's gate is STRUCTURAL and cannot see a change confined to values.
struct SnniWwRegion
{
    void *layer;
    u64 *data;
    u64 size;
    u64 kstart;
    u64 scale;
    u64 sum;        // checksum of the ORIGINAL fill
    bool resident;
};
static std::vector<SnniWwRegion> snni_ww_regions;
static u64 snni_ww_drops = 0, snni_ww_refills = 0, snni_ww_dropped_bytes = 0;

static void snni_ww_reset() { snni_ww_regions.clear(); }

static u64 snni_ww_checksum(const u64 *p, u64 n)
{
    // FNV-1a over the words. Order-sensitive, which is the point: a permuted refill must fail.
    u64 h = 1469598103934665603ULL;
    for (u64 i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static void snni_ww_record(void *layer, void *data, u64 size, u64 kstart, bool /*isw*/, u64 scale)
{
    SnniWwRegion r;
    r.layer = layer; r.data = (u64 *)data; r.size = size;
    r.kstart = kstart; r.scale = scale;
    r.sum = snni_ww_checksum(r.data, size);
    r.resident = true;
    snni_ww_regions.push_back(r);
}

// Only whole pages can be discarded, so the region is rounded INWARD. On a 18,9 MB tensor that
// gives up at most two pages, and rounding outward would discard a neighbour's bytes.
static void snni_ww_drop(SnniWwRegion &r)
{
    if (!r.resident) return;
    const size_t PG = 4096;
    uintptr_t a = (uintptr_t)r.data;
    uintptr_t b = a + r.size * sizeof(u64);
    uintptr_t lo = (a + PG - 1) & ~(uintptr_t)(PG - 1);
    uintptr_t hi = b & ~(uintptr_t)(PG - 1);
    if (hi > lo)
    {
        madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
        snni_ww_dropped_bytes += (u64)(hi - lo);
    }
    r.resident = false;
    ++snni_ww_drops;
}

static void snni_ww_refill(SnniWwRegion &r)
{
    if (r.resident) return;
    u64 k = r.kstart;
    for (u64 i = 0; i < r.size; ++i) r.data[i] = (u64)snni_fixed_value(k++, r.scale, 28000);
    // THE CHECK THE GATE CANNOT DO. A structural gate would pass a wrong value silently.
    u64 s = snni_ww_checksum(r.data, r.size);
    if (s != r.sum)
    {
        fprintf(stderr, "FATAL|s7_weight_window|refill mismatch: region of %lu words at k=%lu "
                        "checksums %llu against the original %llu\\n",
                (unsigned long)r.size, (unsigned long)r.kstart,
                (unsigned long long)s, (unsigned long long)r.sum);
        fflush(stderr);
        abort();
    }
    r.resident = true;
    ++snni_ww_refills;
}

// DIAGNOSTIC, not a lever: are `_MHADummy`'s weights actually costing anything?
//
// The remaining `heap:main <- bert.h:89` row is 226,6 MB, and 226,6/12 = 18,9 MB is exactly
// `wQKV` (n_embed x 3*n_heads*dim_W) plus `wProj` (n_embed x n_embed) per block. That identifies
// the row. What it does NOT establish is that the memory is real: `Tensor2D`'s constructor is
// `data(new T[d1 * d2])` with no initialiser, and `_MHADummy` has no `getweights()` override, so
// the harness fill never reaches these arrays and nothing writes them. Untouched anonymous pages
// are not resident, and a READ of them maps the shared zero page, which a pagemap census can count
// as present while it costs no physical memory at all.
//
// So before a lever is aimed here, `mincore` is asked how many of those pages the kernel actually
// has, and the first words are printed to say whether anything was ever written. One line, no
// allocation, no behaviour change.
static void snni_mha_probe(void *data, u64 elems, const char *what)
{
    const size_t PG = 4096;
    uintptr_t a = (uintptr_t)data, b = a + elems * sizeof(u64);
    uintptr_t lo = (a + PG - 1) & ~(uintptr_t)(PG - 1);
    uintptr_t hi = b & ~(uintptr_t)(PG - 1);
    if (hi <= lo) return;
    size_t pages = (hi - lo) / PG;
    std::vector<unsigned char> vec(pages, 0);
    long rc = mincore((void *)lo, (size_t)(hi - lo), vec.data());
    size_t res = 0;
    if (rc == 0) for (size_t i = 0; i < pages; ++i) res += (vec[i] & 1);
    u64 *p = (u64 *)data;
    int nonzero = 0;
    for (u64 i = 0; i < elems && i < 4096; ++i) if (p[i]) { nonzero = 1; break; }
    fprintf(stderr, "SNNI_MHA|%s|elems=%lu|bytes=%lu|resident_kb=%lu|mincore_rc=%ld|nonzero=%d\\n",
            what, (unsigned long)elems, (unsigned long)(elems * 8),
            (unsigned long)(res * PG / 1024), rc, nonzero);
    fflush(stderr);
}

template <typename T>
static void snni_ww_in(Layer<T> *l)
{
    for (auto &r : snni_ww_regions) if (r.layer == (void *)l) snni_ww_refill(r);
}

template <typename T>
static void snni_ww_out(Layer<T> *l)
{
    for (auto &r : snni_ww_regions) if (r.layer == (void *)l) snni_ww_drop(r);
}

template <typename T>
static void snni_fill_deterministic(SytorchModule<T> *net, Tensor<T> &input, u64 scale)
{
'''

# The hooks are armed after the fill has recorded everything, and the report is printed at exit of
# the dealer phase alongside the other SNNI_ markers.
ARM_OLD = '''    snni_fill_deterministic(net, input, scale);
'''
ARM_NEW = '''    snni_fill_deterministic(net, input, scale);
    // s7_weight_window: arm the hooks only AFTER the fill, so the recording walk itself runs with
    // every tensor resident and the checksums describe the state the baseline had.
    snni_weights_in<u64> = &snni_ww_in<u64>;
    snni_weights_out<u64> = &snni_ww_out<u64>;
    // THE DROP NOW HAPPENS INSIDE THE FILL, one region at a time, and that is the third placement
    // this lever has had. The measurements that forced each move:
    //
    //   run 46282114  armed after the fill, first drop after the first layer's forward, i.e. AFTER
    //                 `new SIGMAKeygen` pinned its buffer.  96 drops, 453,3 MB, peak -0,34%.
    //   run 46284570  all regions dropped up front, before the window exists.  288 drops, 1,36 GB,
    //                 peak -0,25%. Still nothing, and the poller says why: the published peak is a
    //                 spike the 100 ms poller missed by 24% (VmHWM 2.134.240 against 1.621.076),
    //                 i.e. it sits in the first second, DURING the fill, before any drop ran.
    //
    // So the fill itself is the suspect, and a lever that tidies up after it cannot reach it. The
    // record-and-drop is now interleaved with the loop that writes the values, so the program never
    // holds more than the region it is currently filling.
    (void)0;
    // The MHA arrays the fill never reaches, measured rather than assumed.
    topologicalApply(net->root, [&](LayerGraphNode<u64> *node, LayerGraphNode<u64> *_root) {
        auto *m = dynamic_cast<_MHADummy<u64> *>(node->layer);
        if (!m) return;
        snni_mha_probe((void *)m->wQKV.data, m->wQKV.d1 * m->wQKV.d2, "wQKV");
        snni_mha_probe((void *)m->wProj.data, m->wProj.d1 * m->wProj.d2, "wProj");
    });
    {
        u64 tot = 0, mx = 0;
        for (auto &r : snni_ww_regions) { tot += r.size * 8; if (r.size * 8 > mx) mx = r.size * 8; }
        fprintf(stderr, "SNNI_WW|regions=%lu|bytes=%lu|largest=%lu\\n",
                (unsigned long)snni_ww_regions.size(), (unsigned long)tot, (unsigned long)mx);
        fflush(stderr);
    }
'''

# The report sits after the ONLINE phase, not the dealer's: printing at sigmaKeygen->close() reported
# refills=0 (measured too early, before the online pass), which read like a lever that never fired.
REPORT_OLD = '''    sigma->close();
'''
REPORT_NEW = '''    sigma->close();
    fprintf(stderr, "SNNI_WW|drops=%lu|refills=%lu|dropped_bytes=%lu\\n",
            (unsigned long)snni_ww_drops, (unsigned long)snni_ww_refills,
            (unsigned long)snni_ww_dropped_bytes);
    fflush(stderr);
'''


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: patch_weight_window.py <source tree>")
    tree = sys.argv[1]
    hp, cup = os.path.join(tree, H), os.path.join(tree, CU)
    for f in (hp, cup):
        if not os.path.exists(f):
            sys.exit(f"FAILED: {f} does not exist")
    h, cu = open(hp).read(), open(cup).read()

    if "snni_weights_in" in h:
        sys.exit("FAILED: this tree already carries the lever")

    # THE FILL MUST BE THE DETERMINISTIC ONE: the mechanism rests on a layer's bytes being a pure
    # function of its counter; against an entropy-seeded fill it would rewrite different weights unseen.
    if "snni_fixed_value(k++, scale, 28000)" not in cu:
        sys.exit("FAILED: the deterministic fill is not in this tree, so a dropped weight cannot be "
                 "rewritten exactly and the structural gate would not catch the difference")

    for name, old, txt in (("layer hook", HOOK_OLD, h), ("layer declaration", DECL_OLD, h),
                           ("fill counters", FILL_OLD, cu), ("fill record", RECORD_OLD, cu),
                           ("core insertion point", CORE_OLD, cu), ("arm point", ARM_OLD, cu),
                           ("report point", REPORT_OLD, cu)):
        if txt.count(old) != 1:
            sys.exit(f"FAILED: anchor `{name}` matched {txt.count(old)} times, expected exactly 1")

    h = h.replace(DECL_OLD, DECL_NEW, 1).replace(HOOK_OLD, HOOK_NEW, 1)
    cu = cu.replace(CORE_OLD, CORE_NEW, 1).replace(FILL_OLD, FILL_NEW, 1) \
           .replace(RECORD_OLD, RECORD_NEW, 1).replace(ARM_OLD, ARM_NEW, 1) \
           .replace(REPORT_OLD, REPORT_NEW, 1)

    if "#include <sys/mman.h>" not in cu:
        cu = cu.replace("#include <sytorch/module.h>",
                        "#include <sys/mman.h>   // s7_weight_window\n#include <sytorch/module.h>", 1)
    if "#include <vector>" not in cu:
        cu = cu.replace("#include <sys/mman.h>   // s7_weight_window",
                        "#include <sys/mman.h>   // s7_weight_window\n#include <vector>", 1)

    open(hp, "w").write(h)
    open(cup, "w").write(cu)
    print("patch_weight_window: hooks around Layer::forward, per-region drop and checksummed "
          "refill, report at the dealer's close")


if __name__ == "__main__":
    main()
