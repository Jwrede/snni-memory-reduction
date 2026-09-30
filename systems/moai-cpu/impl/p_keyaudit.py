#!/usr/bin/env python3
"""Probe-only key audit MANIFEST.md requires before publishing: count the Galois keys each create_galois_keys call actually produces (target derivation assumes 31 for gal_keys, gal_keys_boot open) plus pool alloc_byte_count per phase. A reduced-batch probe gives identical key counts.

    p_keyaudit.py [/root/moai]        applied AFTER pin_seed_and_gate.py, probe images only
"""
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/moai")
HDR = root / "include" / "test" / "test_full_scheme.hpp"
if not HDR.is_file():
    sys.exit("FAILED: no %s" % HDR)


def edit(path, old, new, what, already, count=1):
    txt = path.read_text()
    if already in txt:
        print("  already patched: %s" % what)
        return
    if txt.count(old) != count:
        sys.exit("FAILED: anchor for %r matched %d times, expected %d" % (what, txt.count(old), count))
    path.write_text(txt.replace(old, new, count))
    print("  patched: %s  (%s)" % (what, path.name))


# ------------------------------------------------------------------------------------------------
# 1. The audit helper, placed after pin_seed_and_gate.py's block so it can use snni_mark.
edit(
    HDR,
    "void all_layer_test(){\n    cout <<\"Task: test BERT in CKKS scheme: \"<<endl;",
    r"""// --- probe only: the key audit MANIFEST.md requires -------------------------------------
// Reports what a key set COST, from the object itself rather than from a derivation: the number of
// key-switching keys it holds, the galois elements it was built for, and SEAL's own serialised
// size. `save_size` is the serialised form and so is a lower bound on residency rather than an
// estimate of it, which is the direction an audit should err in.
static void snni_key_audit(const char *name, const seal::GaloisKeys &k) {
    size_t n = 0;
    for (const auto &v : k.data()) if (!v.empty()) n++;
    fprintf(stderr, "KEYAUDIT|%s|keys=%zu|save_bytes=%zu\n",
            name, n, (size_t)k.save_size(seal::compr_mode_type::none));
    fflush(stderr);
}
static void snni_pool_mark(const char *name) {
    fprintf(stderr, "POOL|%s|alloc_bytes=%zu\n",
            name, (size_t)seal::MemoryManager::GetPool().alloc_byte_count());
    fflush(stderr);
    snni_mark(name);
}
// ---------------------------------------------------------------------------------------

void all_layer_test(){
    cout <<"Task: test BERT in CKKS scheme: "<<endl;""",
    "key audit helpers",
    already="KEYAUDIT|%s|keys=",
)

# ------------------------------------------------------------------------------------------------
# 2. The two key sets, each audited where finished: gal_keys (assumed 31) and gal_keys_boot (open).
edit(
    HDR,
    "    keygen.create_galois_keys(gal_keys);",
    r"""    snni_pool_mark("before_gal_keys");
    keygen.create_galois_keys(gal_keys);
    snni_key_audit("gal_keys", gal_keys);
    snni_pool_mark("after_gal_keys");""",
    "audit around the default galois key set",
    already='snni_key_audit("gal_keys"',
)

edit(
    HDR,
    "    keygen.create_galois_keys(gal_steps_vector, gal_keys_boot);",
    r"""    fprintf(stderr, "KEYAUDIT|gal_steps_vector|requested=%zu\n", gal_steps_vector.size());
    fflush(stderr);
    snni_pool_mark("before_gal_keys_boot");
    keygen.create_galois_keys(gal_steps_vector, gal_keys_boot);
    snni_key_audit("gal_keys_boot", gal_keys_boot);
    snni_pool_mark("after_gal_keys_boot");""",
    "audit around the bootstrap galois key set",
    already='snni_key_audit("gal_keys_boot"',
)

# ------------------------------------------------------------------------------------------------
# 3. The batch as a knob (MOAI_NUM_X) so the probe is cheap; num_X is rewritten to read the env at
# static-init time, keeping dependent file-scope arrays sized. Default 256 (upstream's value).
edit(
    HDR,
    "const int num_X = 256;",
    r"""static int snni_num_X_init() {
    const char *v = getenv("MOAI_NUM_X");
    int n = (v && *v) ? atoi(v) : 0;
    if (n <= 0) n = 256;
    fprintf(stderr, "SNNI_PARAM|MOAI_NUM_X=%d|source=%s\n", n, (v && *v && n != 256) ? "env" : "default");
    fflush(stderr);
    return n;
}
const int num_X = snni_num_X_init();""",
    "batch size from MOAI_NUM_X",
    already='snni_num_X_init',
)

# ------------------------------------------------------------------------------------------------
# The audit needs the pool and key headers; the file may not include them directly.
txt = HDR.read_text()
if "#include <seal/seal.h>" not in txt and "seal/seal.h" not in txt:
    print("  NOTE: seal/seal.h not included directly in %s; relying on include.hpp" % HDR.name)

print("\n--- every call site the audit now brackets ---")
hits = subprocess.run(
    ["grep", "-n", "snni_pool_mark\\|snni_key_audit\\|snni_num_X_init", str(HDR)],
    capture_output=True, text=True).stdout.strip()
print(hits if hits else "(none -- the patch did not take)")
print("--- end ---\n")
print("PATCH_OK" if hits else "PATCH_FAIL")
sys.exit(0 if hits else 3)
