#!/usr/bin/env python3
"""Give MOAI-CPU a seed, a gate observable and a declared depth in its own source tree. Run inside the image build on a pristine MOAI tree before cmake; this is the measured system, applied identically to baseline and every step.

The shipped program has NO gate (all decryptions are commented out); the fix uses the live per-layer layer_<id>.txt output, pins SEAL's PRNG, and declares depth via MOAI_N_LAYERS (default 12, measured=2).
"""
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/moai")
HDR = "include/test/test_full_scheme.hpp"


def edit(path, old, new, what, already, count=1):
    """Apply one patch, idempotently. `already` must be a string that exists ONLY afterwards."""
    p = root / path
    src = p.read_text()
    if already in src:
        print(f"  already patched: {what}")
        return
    if src.count(old) < count:
        sys.exit(f"FAILED: anchor for '{what}' appears {src.count(old)} times in {path}, "
                 f"expected {count}. Upstream moved; re-derive the patch rather than loosen it.")
    p.write_text(src.replace(old, new, count))
    if already not in p.read_text():
        sys.exit(f"FAILED: applied '{what}' but its marker is absent from {path}.")
    print(f"  patched: {what}  ({path})")


# --------------------------------------------------------------------------------------------
# 0. One helper block, so the knob, the seed and the markers cannot drift apart.
edit(
    HDR,
    "void all_layer_test(){",
    r"""// ---------------------------------------------------------------------------------------
// SNNI campaign instrumentation. Not a lever: this is part of the measured system and is applied
// identically to the baseline and to every step.
//
// `cstdio` is included EXPLICITLY rather than relied on. `fprintf` and `sscanf` already compiled
// here through some transitive include, which is exactly the kind of dependency that breaks on an
// upstream header change and then reads as a mysterious build failure hours later.
#include <cstdio>
static int snni_int_env(const char *name, int dflt) {
    const char *v = getenv(name);
    if (!v || !*v) {
        fprintf(stderr, "SNNI_PARAM|%s=%d|source=default\n", name, dflt);
        fflush(stderr);
        return dflt;
    }
    int n = atoi(v);
    if (n <= 0) {
        fprintf(stderr, "SNNI_PARAM|%s=%d|source=default (unparsable %s)\n", name, dflt, v);
        fflush(stderr);
        return dflt;
    }
    fprintf(stderr, "SNNI_PARAM|%s=%d|source=env\n", name, n);
    fflush(stderr);
    return n;
}

// A named point in the run, with the kernel's own high-water mark beside the current RSS. VmHWM is
// monotone, so the interval in which it last rose is exactly the interval the peak fell in -- which
// is how the GPU sibling's peak was bracketed without any sampling risk. Cheap: two lines of
// /proc per call.
static void snni_mark(const char *name) {
    long rss = 0, hwm = 0;
    FILE *f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (!strncmp(line, "VmRSS:", 6)) sscanf(line + 6, "%ld", &rss);
            else if (!strncmp(line, "VmHWM:", 6)) sscanf(line + 6, "%ld", &hwm);
        }
        fclose(f);
    }
    fprintf(stderr, "MEM|%s|rss_kb=%ld|hwm_kb=%ld\n", name, rss, hwm);
    fflush(stderr);
}
// ---------------------------------------------------------------------------------------

void all_layer_test(){""",
    "instrumentation helpers",
    already="SNNI_PARAM|%s=%d|source=default",
)

# --------------------------------------------------------------------------------------------
# 1. The depth, as a declared runtime parameter (default upstream's 12, so an unset var never
# quietly shortens a run).
edit(
    HDR,
    "for (int layer_id = 0; layer_id < 12; ++layer_id){",
    r"""int snni_n_layers = snni_int_env("MOAI_N_LAYERS", 12);
if (snni_n_layers > 12) snni_n_layers = 12;
fprintf(stderr, "WORKLOAD|model=bert-base|layers=%d|of=12|num_X=%d|num_row=%d|num_col=%d\n",
        snni_n_layers, num_X, num_row, num_col);
fflush(stderr);
snni_mark("run_begin");
for (int layer_id = 0; layer_id < snni_n_layers; ++layer_id){""",
    "layer depth from MOAI_N_LAYERS",
    already='snni_int_env("MOAI_N_LAYERS"',
)

# --------------------------------------------------------------------------------------------
# 2. SEAL's PRNG (secret/public/Galois keys and every encryption's noise); without it no two runs
# are comparable.
edit(
    HDR,
    "    SEALContext context(parms, true, sec_level_type::none);",
    r"""    // SNNI: pin SEAL's randomness before the context is built. Everything the keys and the
    // encryptions draw comes from this factory; leaving it unset means a fresh secret key per run,
    // and then a step's output cannot be compared with the baseline's even in principle.
    {
        const char *snni_seed_env = getenv("SNNI_SEED");
        if (snni_seed_env != nullptr) {
            uint64_t snni_seed = strtoull(snni_seed_env, nullptr, 0);
            seal::prng_seed_type snni_seal_seed{};
            for (size_t i = 0; i < snni_seal_seed.size(); i++) {
                snni_seal_seed[i] = snni_seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1);
            }
            parms.set_random_generator(
                std::make_shared<seal::Blake2xbPRNGFactory>(snni_seal_seed));
            fprintf(stderr, "SEEDED|seal_prng|seed=0x%llx\n", (unsigned long long)snni_seed);
            fflush(stderr);
        }
    }
    SEALContext context(parms, true, sec_level_type::none);""",
    "SEAL PRNG factory",
    already="SEEDED|seal_prng|",
)

# --------------------------------------------------------------------------------------------
# 3. The gate already exists upstream: all_layer_test writes each layer's decrypted output to
# layer_<id>.txt, which the harness flattens into logs/gate.txt. So no decryption is added here;
# the program writes into its CURRENT DIRECTORY, which the runner must make writable.

# Only cstdio is missing (include.hpp already has cstdlib, cstring, cmath, memory); guard on that
# one, not a header already present.
edit(
    "include/include.hpp",
    "#include <iostream>",
    """#include <iostream>
#include <cstdio>""",
    "include.hpp gains cstdio for the instrumentation",
    already="#include <cstdio>",
)

# --------------------------------------------------------------------------------------------
# 4. The workload: upstream's test.cpp runs three unit tests before all_layer_test, each building
# its own SEAL context/keys, so measuring them together would attribute their allocations to BERT.
# Make only all_layer_test live.
test_cpp = (root / "test.cpp").read_text()
if "SNNI_WORKLOAD_SELECTED" not in test_cpp:
    (root / "test.cpp").write_text("""// SNNI: only the measured workload is live. Upstream also called batch_input_test,
// ct_pt_matrix_mul_test and ct_ct_matrix_mul_test before it, each building its own SEAL context
// and keys; measuring them together would attribute their allocations to BERT.
// SNNI_WORKLOAD_SELECTED
#include <iostream>
#include "include.hpp"

using namespace std;

int main(){
    cout << "----------------------FULL SCHEME------------------" << endl;
    all_layer_test();
    return 0;
}
""")
    print("  patched: test.cpp now calls only all_layer_test  (test.cpp)")
else:
    print("  already patched: workload selection")

# --------------------------------------------------------------------------------------------
# 5. Phase markers, so the conservation rule (a reduction only helps at or after the last growth
# event) has a timeline to apply to. Names are MOAI-GPU's, deliberately, so the CPU/GPU pair's
# reservation profiles are comparable phase by phase. Each anchor is a unique line upstream already
# prints (distinct timing variables), so a moved anchor fails the build rather than marking wrong.
_PHASES = [
    ('cout << "preparing bootstrapping..." << endl;',              "keys_created",       "before"),
    ("    bootstrapper.slot_vec.push_back(logn);",                 "boot_keys_created",  "after"),
    ('cout <<"Modulus chain index before attention block: "',      "before_attention",   "before"),
    ('cout <<"Attention block time = "<<att_block_time<<endl;',    "after_attention",    "after"),
    ('cout <<"selfoutput time = "<<selfoutput_time<<endl;',        "after_selfoutput",   "after"),
    ('cout <<"bootstrapping time = "<<boot_time<<endl;',           "after_bootstrap1",   "after"),
    ('cout <<"layernorm time = "<<layernorm_time<<endl;',          "after_layernorm1",   "after"),
    ('cout <<"bootstrapping time = "<<boot_time2<<endl;',          "after_bootstrap2",   "after"),
    ('cout <<"Inter layer time = "<<inter_time<<endl;',            "after_intermediate", "after"),
    ('cout <<"gelu time = "<<gelu_time<<endl;',                    "after_gelu",         "after"),
    ('cout <<"Final layer time = "<<final_time<<endl;',            "after_final",        "after"),
    ('cout <<"bootstrapping time = "<<boot_time3<<endl;',          "after_bootstrap3",   "after"),
    ('cout <<"layernorm time = "<<layernorm_time2<<endl;',         "after_layernorm2",   "after"),
    ('cout <<"bootstrapping time = "<<boot_time4<<endl;',          "after_bootstrap4",   "after"),
]

for _anchor, _name, _where in _PHASES:
    _call = 'snni_mark("%s");' % _name
    _new = (_call + "\n    " + _anchor) if _where == "before" else (_anchor + "\n    " + _call)
    edit(HDR, _anchor, _new, "phase marker " + _name, already=_call)

# The layer boundary carries its index (the run does two layers); the only marker whose name is
# built at runtime.
edit(
    HDR,
    'cout <<"---------------Layer No. "<<layer_id<<"-------------------"<<endl;',
    'cout <<"---------------Layer No. "<<layer_id<<"-------------------"<<endl;\n'
    '    { char _snni_lb[48]; snprintf(_snni_lb, sizeof(_snni_lb), "layer_%d_begin", layer_id);\n'
    '      snni_mark(_snni_lb); }',
    "layer boundary marker",
    already="_snni_lb",
)


print("\n--- randomness and clock reads left in the measured tree ---")
hits = subprocess.run(
    ["grep", "-rnE", r"random_device|mt19937|srand\(|\brand\(\)|KeyGenerator",
     "--include=*.hpp", "--include=*.cpp", str(root / "include"), str(root / "test.cpp")],
    capture_output=True, text=True).stdout.strip()
print(hits if hits else "(none)")
print("--- end ---\n")
print("PATCH_OK")
