#!/usr/bin/env python3
"""Probe-only attention-instant target micro-benchmark: only the protocol-required set at one head's
QK^T instant, T=1, resident footprint (VmRSS) as one integrated measurement (target: attention keys +
one head's Q, K, V, scores + relin, context excluded).

Companion of p_floor_bench.py (bootstrap instant). Body inserted after create_relin_keys, before the
30-key and 47-key banks: VmRSS = context + relin + 14 rotation keys + one head's blocks. 14 keys stand
for the attention's minimal set (m2_minimal_att_keys); key size is rotation-independent.

    p_floor_bench_att.py [/root/moai]   applied to a fresh pinned checkout, probe image only, the
                                        same slot pin_seed_and_gate.py / p_keyaudit.py occupy.

Run with SNNI_FLOOR_ATT=1 (and OMP_NUM_THREADS=1).
"""
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


# 1. helpers, before all_layer_test at global scope. FLOOR_ATT lines to stderr only.
edit(
    HDR,
    "void all_layer_test(){\n    cout <<\"Task: test BERT in CKKS scheme: \"<<endl;",
    r"""// --- probe only: the ATTENTION-instant floor micro-benchmark --------------------------
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <exception>
#ifdef _OPENMP
#include <omp.h>
#endif

static int snni_att_enabled() {
    static int v = -1;
    if (v < 0) { const char *e = getenv("SNNI_FLOOR_ATT"); v = (e && *e) ? 1 : 0; }
    return v;
}
static std::map<std::string, long> &snni_att_map() { static std::map<std::string, long> m; return m; }
static void snni_att_status(long &rss_kb, long &hwm_kb) {
    rss_kb = -1; hwm_kb = -1;
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (!strncmp(line, "VmRSS:", 6))  sscanf(line + 6,  "%ld", &rss_kb);
        else if (!strncmp(line, "VmHWM:", 6)) sscanf(line + 6, "%ld", &hwm_kb);
    }
    fclose(f);
}
static void snni_att(const char *name) {
    if (!snni_att_enabled()) return;
    long rss = -1, hwm = -1;
    snni_att_status(rss, hwm);
    snni_att_map()[std::string(name)] = rss;
    fprintf(stderr, "FLOOR_ATT|%s|vmrss_kb=%ld|vmhwm_kb=%ld\n", name, rss, hwm);
    fflush(stderr);
}
static long snni_att_get(const char *name) {
    std::map<std::string, long> &m = snni_att_map();
    std::map<std::string, long>::iterator it = m.find(std::string(name));
    return (it == m.end()) ? 0 : it->second;
}
static void snni_att_touch(const seal::Ciphertext &ct) {
    const uint64_t *p = ct.data();
    size_t n = (size_t)ct.size() * (size_t)ct.poly_modulus_degree() * (size_t)ct.coeff_modulus_size();
    volatile uint64_t s = 0;
    for (size_t i = 0; i < n; ++i) s ^= p[i];
    (void)s;
}
// Encrypt zero at top level, mod-switch down to target_limbs RNS limbs, touch it resident.
// A ciphertext at chain index c has c+1 limbs: Q,K at 13 -> 14 limbs; scores at 12 -> 13; V at 2 -> 3.
static seal::Ciphertext snni_att_ct(seal::SEALContext &context, seal::CKKSEncoder &encoder,
        seal::Encryptor &encryptor, seal::Evaluator &evaluator, double scale, size_t target_limbs) {
    (void)context;
    seal::Plaintext pt;
    encoder.encode(0.0, scale, pt);
    seal::Ciphertext ct;
    encryptor.encrypt(pt, ct);
    while (ct.coeff_modulus_size() > target_limbs) evaluator.mod_switch_to_next_inplace(ct);
    snni_att_touch(ct);
    return ct;
}
// ---------------------------------------------------------------------------------------

void all_layer_test(){
    cout <<"Task: test BERT in CKKS scheme: "<<endl;
    if (snni_att_enabled()) {
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        fprintf(stderr, "FLOOR_ATT|begin|note=T=1 attention-instant resident floor; deltas are VmRSS\n");
        fflush(stderr);
        snni_att("after_start");
    }""",
    "attention floor-bench helpers",
    already="FLOOR_ATT|%s|vmrss_kb=",
)

# 2a. baseline read right after create_public_key, BEFORE relin, so the relin key counts in target.
edit(
    HDR,
    "    keygen.create_public_key(public_key);",
    "    keygen.create_public_key(public_key);\n    snni_att(\"ctx_pub\");",
    "ctx_pub read",
    already='snni_att("ctx_pub")',
)

# 2. the body, right after create_relin_keys: context + public + relin are resident, no key bank yet.
edit(
    HDR,
    "    keygen.create_relin_keys(relin_keys);",
    r"""    keygen.create_relin_keys(relin_keys);
    if (snni_att_enabled()) {
        // ATTENTION-INSTANT floor. Local encoder/encryptor/evaluator (public_key already built);
        // the 30-key attention bank and the 47-key bootstrap bank are NOT built yet, so VmRSS at
        // att_instant is exactly context + relin + 14 rotation keys + one head's blocks.
        seal::CKKSEncoder fenc(context);
        seal::Encryptor fencr(context, public_key);
        seal::Evaluator fev(context, fenc);
        snni_att("att_context");            // context + public + relin, the baseline to subtract
        // 14 attention rotation keys (14 key-switching keys; resident size is rotation-independent,
        // so distinct steps 1..14 give the same 14 x 1.32 GB as the minimal attention set of m2).
        std::vector<int> steps; for (int i = 1; i <= 14; ++i) steps.push_back(i);
        seal::GaloisKeys att_keys;
        keygen.create_galois_keys(steps, att_keys);
        snni_att("att_after_keys");
        // one head: Q, K and one thread's rotated key copy at chain 13 (14 limbs), scores at 12
        // (13 limbs), V at 2 (3 limbs). col_W = 64 per head, scores 128. A right-sized ciphertext
        // is built once per level as a TEMPLATE (encrypt at the top level, mod-switch down, copy to a
        // tight allocation) and the vectors are filled by copy, so only level-size pages are touched;
        // building each entry by encrypt+down-switch would keep the top-level pages resident, the
        // capacity inflation the reduction's ct_capacity_shrink step removes, and would mismeasure.
        seal::Ciphertext t14 = snni_att_ct(context, fenc, fencr, fev, scale, 14);
        seal::Ciphertext t13 = snni_att_ct(context, fenc, fencr, fev, scale, 13);
        seal::Ciphertext t3  = snni_att_ct(context, fenc, fencr, fev, scale, 3);
        std::vector<seal::Ciphertext> blocks;
        blocks.reserve(64*3 + 128 + 64);
        for (int i = 0; i < 192; ++i) { blocks.push_back(t14); snni_att_touch(blocks.back()); }  // Q,K,copy
        for (int i = 0; i < 128; ++i) { blocks.push_back(t13); snni_att_touch(blocks.back()); }  // scores
        for (int i = 0; i < 64;  ++i) { blocks.push_back(t3);  snni_att_touch(blocks.back()); }  // V
        snni_att("att_instant");
        long ctxp  = snni_att_get("ctx_pub");     // context + public key, before relin
        long total = snni_att_get("att_instant");
        long keys  = snni_att_get("att_after_keys") - snni_att_get("att_context");
        fprintf(stderr,
            "FLOOR_ATT_SUMMARY|ctx_pub_kb=%ld|keys14_kb=%ld|blocks_kb=%ld|instant_kb=%ld|target_kb=%ld\n",
            ctxp, keys, total - snni_att_get("att_after_keys"), total, total - ctxp);
        fflush(stderr);
        return;   // nothing else is allocated; templates and blocks stay resident through the reads
    }""",
    "attention floor-bench body",
    already='snni_att("att_context")',
)

print("\n--- attention floor-bench sites now in the source ---")
hits = subprocess.run(
    ["grep", "-n", "snni_att\\|FLOOR_ATT", str(HDR)],
    capture_output=True, text=True).stdout.strip()
print(hits if hits else "(none -- the patch did not take)")
_txt = HDR.read_text()
ok = bool(hits) and 'snni_att("att_context")' in _txt and 'snni_att("att_instant")' in _txt
print("PATCH_OK" if ok else "PATCH_FAIL")
sys.exit(0 if ok else 3)
