#!/usr/bin/env python3
"""Measurement-only instrumentation of the pristine MOAI-GPU source (instrument.py <source-root>).

Adds a dual-pool phase marker (`MEM| <tag> | ...`) at fifteen phase boundaries and one `GPUPEAK|`
line from the pool's driver-maintained high-water attributes. Same at baseline and every step; not
a lever. Edits in place, refuses a second application.
"""
import re
import sys

MARKER_IMPL = r'''
// ---------------------------------------------------------------------------------------------
// SNNI campaign instrumentation. MEASUREMENT ONLY: reads counters, allocates nothing, changes no
// control flow. See systems/moai-gpu/impl/instrument.py for why it exists and what it may not do.
static inline void snni_read_host_rss_kb(long &rss_kb, long &hwm_kb)
{
    rss_kb = -1; hwm_kb = -1;
    std::ifstream st("/proc/self/status");
    std::string line;
    while (std::getline(st, line)) {
        if (line.rfind("VmRSS:", 0) == 0)      rss_kb = std::atol(line.c_str() + 6);
        else if (line.rfind("VmHWM:", 0) == 0) hwm_kb = std::atol(line.c_str() + 6);
    }
}

static inline long long snni_epoch_ms_now()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

// Live and reserved bytes of the default device memory pool, which IS Phantom's allocator: it
// allocates through cudaMallocAsync throughout (cuda_wrapper.cuh make_cuda_auto_ptr).
// used = currently live, reserved = what the pool holds including retained-but-freed blocks.
// The gap between them is retained-dead memory, and it is a finding rather than an error: this
// pool deliberately does not release, and trimming it measured WORSE in the retired campaign.
static inline void snni_pool_now(unsigned long long &used, unsigned long long &reserved)
{
    used = 0ULL; reserved = 0ULL;
    int dev = 0; cudaGetDevice(&dev);
    cudaMemPool_t mp;
    if (cudaDeviceGetDefaultMemPool(&mp, dev) == cudaSuccess) {
        cudaMemPoolGetAttribute(mp, cudaMemPoolAttrUsedMemCurrent, &used);
        cudaMemPoolGetAttribute(mp, cudaMemPoolAttrReservedMemCurrent, &reserved);
    }
}

// The pool's own high-water marks. Driver-maintained, so they cannot be missed by a sampler.
static inline void snni_pool_high(unsigned long long &used_hi, unsigned long long &reserved_hi)
{
    used_hi = 0ULL; reserved_hi = 0ULL;
    int dev = 0; cudaGetDevice(&dev);
    cudaMemPool_t mp;
    if (cudaDeviceGetDefaultMemPool(&mp, dev) == cudaSuccess) {
        cudaMemPoolGetAttribute(mp, cudaMemPoolAttrUsedMemHigh, &used_hi);
        cudaMemPoolGetAttribute(mp, cudaMemPoolAttrReservedMemHigh, &reserved_hi);
    }
}

static inline void snni_mem_marker(const char *tag)
{
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const double MB = 1024.0 * 1024.0;
    long rss_kb = -1, hwm_kb = -1;
    snni_read_host_rss_kb(rss_kb, hwm_kb);
    unsigned long long pool_used = 0ULL, pool_reserved = 0ULL;
    snni_pool_now(pool_used, pool_reserved);
    std::cout << "MEM| " << tag
              << " | vram_used_MiB: " << (double)(total_b - free_b) / MB
              << " | pool_used_MiB: " << (double)pool_used / MB
              << " | pool_reserved_MiB: " << (double)pool_reserved / MB
              << " | host_rss_MiB: " << (rss_kb < 0 ? -1.0 : rss_kb / 1024.0)
              << " | host_hwm_MiB: " << (hwm_kb < 0 ? -1.0 : hwm_kb / 1024.0)
              << " | ts_ms: " << snni_epoch_ms_now()
              << " | (Free: " << free_b / MB << " MB, Total: " << total_b / MB << " MB)"
              << std::endl;
}

// The harness contract. make_steps.py takes the device peak from this line and does not care who
// wrote it (device_peak(), make_steps.py:284), so a non-torch system supplies it here.
static inline void snni_gpupeak(int party)
{
    unsigned long long used_hi = 0ULL, reserved_hi = 0ULL;
    snni_pool_high(used_hi, reserved_hi);
    long rss_kb = -1, hwm_kb = -1;
    snni_read_host_rss_kb(rss_kb, hwm_kb);
    std::cout << "GPUPEAK|party=" << party
              << "|peak_alloc_kb=" << (used_hi / 1024ULL)
              << "|peak_reserved_kb=" << (reserved_hi / 1024ULL)
              << std::endl;
    std::cout << "HOSTPEAK|party=" << party << "|vmhwm_kb=" << hwm_kb << std::endl;
}
// ---------------------------------------------------------------------------------------------
'''

# (anchor substring, which occurrence, marker tag). The occurrence index is explicit because four
# anchors are the same sentence printed about different arrays; a wrong pick would place a
# correctly-named marker at the wrong instant. Every entry is asserted below.
MARKERS = [
    ('Galois key generated from steps vector.',        1, 'keys_created'),
    ('generate_LT_coefficient_3();',                   1, 'bootstrapping_prepared'),
    ('Modulus chain index before attention block:',    1, 'inputs_prepared'),
    ('Attention block time = ',                        1, 'after_attention'),
    ('Modulus chain index for the result:',            1, 'after_selfoutput'),
    ('Modulus chain index after bootstrapping:',       1, 'after_bootstrap1'),
    ('Modulus chain index after layernorm:',           1, 'after_layernorm1'),
    ('Modulus chain index after bootstrapping:',       2, 'after_bootstrap2'),
    ('Modulus chain index before intermediate linear:', 1, 'before_intermediate'),
    ('Modulus chain index after inter layer:',         1, 'after_intermediate'),
    ('Modulus chain index for gelu:',                  1, 'after_gelu'),
    ('Modulus chain index after final layer:',         1, 'after_final'),
    ('Modulus chain index after bootstrapping:',       3, 'after_bootstrap3'),
    ('Modulus chain index after layernorm:',           2, 'after_layernorm2'),
    ('Modulus chain index after bootstrapping:',       4, 'after_bootstrap4'),
]


def fail(msg):
    sys.exit(f"instrument.py: {msg}")


def patch_include(path):
    src = open(path).read()
    if 'snni_mem_marker' in src:
        fail(f"{path} is already instrumented, refusing to apply twice")
    anchor = 'static inline void print_cuda_meminfo(const char *tag)'
    if anchor not in src:
        fail(f"anchor for the marker block not found in {path}")
    src = src.replace(anchor, MARKER_IMPL.lstrip('\n') + '\n' + anchor, 1)
    open(path, 'w').write(src)
    print(f"  include.cuh: marker implementation inserted")


def patch_test(path):
    lines = open(path).read().splitlines(keepends=True)
    if any('snni_mem_marker' in ln for ln in lines):
        fail(f"{path} is already instrumented, refusing to apply twice")

    # Resolve every anchor to a line index first, so a miss aborts before anything is written.
    seen = {}
    plan = []
    for anchor, want_occ, tag in MARKERS:
        idx = None
        occ = 0
        for i, ln in enumerate(lines):
            if anchor in ln and 'snni_mem_marker' not in ln:
                # a commented-out line is not a call site
                if ln.lstrip().startswith('//'):
                    continue
                occ += 1
                if occ == want_occ:
                    idx = i
                    break
        if idx is None:
            fail(f"anchor {anchor!r} occurrence {want_occ} (for {tag}) not found; "
                 "the source is not the state this script was written against")
        if idx in seen:
            fail(f"anchor for {tag} resolved to the same line as {seen[idx]}")
        seen[idx] = tag
        plan.append((idx, tag))

    # `inputs_prepared` is emitted twice (also as `before_attention`, nothing happens between them),
    # so the two campaigns' marker sets line up when compared.
    extra = [(idx, 'before_attention') for idx, tag in plan if tag == 'inputs_prepared']
    plan.extend(extra)

    for idx, tag in sorted(plan, key=lambda p: -p[0]):
        indent = re.match(r'[ \t]*', lines[idx]).group(0)
        lines.insert(idx + 1, f'{indent}snni_mem_marker("{tag}");\n')

    # The final GPUPEAK line goes before the closing brace of the function.
    text = ''.join(lines)
    tail = text.rstrip()
    if not tail.endswith('}'):
        fail("test_single_layer.cuh does not end in a closing brace; cannot place GPUPEAK")
    cut = text.rstrip().rfind('}')
    text = text[:cut] + '    snni_gpupeak(0);\n' + text[cut:]
    open(path, 'w').write(text)
    print(f"  test_single_layer.cuh: {len(plan)} markers + GPUPEAK placed")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = sys.argv[1].rstrip('/')
    inc = f"{root}/src/include/include.cuh"
    tst = f"{root}/src/include/test/test_single_layer.cuh"
    print(f"instrumenting {root}")
    patch_include(inc)
    patch_test(tst)
    print("done. measurement only: no allocation, no control flow, no lever.")


if __name__ == '__main__':
    main()
