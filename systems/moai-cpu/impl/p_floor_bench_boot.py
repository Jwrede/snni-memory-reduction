#!/usr/bin/env python3
"""Probe-only BOOTSTRAP-instant target micro-benchmark. Builds the set that sets the MOAI target
(MANIFEST "bootstrap, chunk 128": largest bootstrap key stage of 15 keys + the chunk in flight, 128
ciphertexts at 32 limbs + relin key + one thread's working set, context excluded, 25,267,200 kB)
as ONE integrated set at T=1 and reports its RESIDENT footprint (VmRSS from /proc/self/status).

Companion to p_floor_bench_att.py (the QK^T instant). Like it, the body goes right after
create_relin_keys, BEFORE the 30-key attention bank and the 47-key bootstrap bank are built, so VmRSS
at the reads is context + relin + the stage + the chunk, with no key bank swamping it. The stage is
built as 15 key-switching keys, steps {0, 1, ..., 14}: step 0 is SEAL's conjugation key, the largest
CoeffToSlot/SlotToCoeff stage holds 14 rotations and one conjugation; a galois key's resident size is
rotation-independent. The chunk is filled by copying a right-sized template (encrypt at the top level,
mod-switch down to 32 limbs, copy), as in p_floor_bench_att.py. One thread's working set is the
target's fourth term, built the same way as p_floor_bench.py builds it: a modraised ciphertext at 35
limbs plus the 12 Paterson-Stockmeyer temporaries of the modular reduction (baby[8] + giant[3] + tmp)
at 32 limbs, 35 + 12 x 32 MiB = 429,056 kB. After the instant is read, ONE key switch (a rotation of
one chunk ciphertext with a key of the stage) runs at T=1 and its VmRSS step is reported separately.

    p_floor_bench_boot.py [/root/moai]  applied to a pinned checkout (8bcb0ea), probe image only.

Run the built binary with SNNI_FLOOR_BOOT=1 (and OMP_NUM_THREADS=1).
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


# 1. helpers, before all_layer_test at global scope. FLOOR_BOOT lines to stderr only.
edit(
    HDR,
    "void all_layer_test(){\n    cout <<\"Task: test BERT in CKKS scheme: \"<<endl;",
    r"""// --- probe only: the BOOTSTRAP-instant target micro-benchmark --------------------------
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

static int snni_boot_enabled() {
    static int v = -1;
    if (v < 0) { const char *e = getenv("SNNI_FLOOR_BOOT"); v = (e && *e) ? 1 : 0; }
    return v;
}
static std::map<std::string, long> &snni_boot_map() { static std::map<std::string, long> m; return m; }
static void snni_boot_status(long &rss_kb, long &hwm_kb) {
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
static void snni_boot(const char *name) {
    if (!snni_boot_enabled()) return;
    long rss = -1, hwm = -1;
    snni_boot_status(rss, hwm);
    snni_boot_map()[std::string(name)] = rss;
    fprintf(stderr, "FLOOR_BOOT|%s|vmrss_kb=%ld|vmhwm_kb=%ld\n", name, rss, hwm);
    fflush(stderr);
}
static long snni_boot_get(const char *name) {
    std::map<std::string, long> &m = snni_boot_map();
    std::map<std::string, long>::iterator it = m.find(std::string(name));
    return (it == m.end()) ? 0 : it->second;
}
static void snni_boot_touch(const seal::Ciphertext &ct) {
    const uint64_t *p = ct.data();
    size_t n = (size_t)ct.size() * (size_t)ct.poly_modulus_degree() * (size_t)ct.coeff_modulus_size();
    volatile uint64_t s = 0;
    for (size_t i = 0; i < n; ++i) s ^= p[i];
    (void)s;
}
// Encrypt zero at top level, mod-switch down to target_limbs RNS limbs, touch it resident.
// A ciphertext at chain index c has c+1 limbs: the chunk at chain index 31 -> 32 limbs.
static seal::Ciphertext snni_boot_ct(seal::SEALContext &context, seal::CKKSEncoder &encoder,
        seal::Encryptor &encryptor, seal::Evaluator &evaluator, double scale, size_t target_limbs) {
    (void)context;
    seal::Plaintext pt;
    encoder.encode(0.0, scale, pt);
    seal::Ciphertext ct;
    encryptor.encrypt(pt, ct);
    while (ct.coeff_modulus_size() > target_limbs) evaluator.mod_switch_to_next_inplace(ct);
    snni_boot_touch(ct);
    return ct;
}
// ---------------------------------------------------------------------------------------

void all_layer_test(){
    cout <<"Task: test BERT in CKKS scheme: "<<endl;
    if (snni_boot_enabled()) {
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        fprintf(stderr, "FLOOR_BOOT|begin|note=T=1 bootstrap-instant resident target; deltas are VmRSS\n");
        fflush(stderr);
        snni_boot("after_start");
    }""",
    "bootstrap floor-bench helpers",
    already="FLOOR_BOOT|%s|vmrss_kb=",
)

# 2a. baseline read right after create_public_key, BEFORE relin, so the relin key counts in target.
edit(
    HDR,
    "    keygen.create_public_key(public_key);",
    "    keygen.create_public_key(public_key);\n    snni_boot(\"ctx_pub\");",
    "ctx_pub read",
    already='snni_boot("ctx_pub")',
)

# 2. the body, right after create_relin_keys: context + public + relin are resident, no key bank yet.
edit(
    HDR,
    "    keygen.create_relin_keys(relin_keys);",
    r"""    keygen.create_relin_keys(relin_keys);
    if (snni_boot_enabled()) {
        // BOOTSTRAP-INSTANT target set. Local encoder/encryptor/evaluator (public_key already built);
        // the 30-key attention bank and the 47-key bootstrap bank are NOT built yet.
        seal::CKKSEncoder fenc(context);
        seal::Encryptor fencr(context, public_key);
        seal::Evaluator fev(context, fenc);
        snni_boot("boot_context");          // context + public + relin
        // largest bootstrap key stage: 14 rotations + conjugation (step 0), 15 key-switching keys
        std::vector<int> steps; steps.push_back(0); for (int i = 1; i <= 14; ++i) steps.push_back(i);
        seal::GaloisKeys stage_keys;
        keygen.create_galois_keys(steps, stage_keys);
        fprintf(stderr, "FLOOR_BOOT|stage_keys|count=%zu\n", stage_keys.size()); fflush(stderr);
        snni_boot("boot_after_stage");
        // the chunk in flight: 128 ciphertexts at 32 limbs (chain index 31), built by template copy
        seal::Ciphertext t32 = snni_boot_ct(context, fenc, fencr, fev, scale, 32);
        std::vector<seal::Ciphertext> chunk;
        chunk.reserve(128);
        for (int i = 0; i < 128; ++i) { chunk.push_back(t32); snni_boot_touch(chunk.back()); }
        fprintf(stderr, "FLOOR_BOOT|chunk|count=%zu|limbs=%zu\n", chunk.size(), chunk[0].coeff_modulus_size());
        fflush(stderr);
        snni_boot("boot_after_chunk");
        // one thread's working set: a modraised ciphertext at 35 limbs + 12 PS temporaries at 32 limbs
        seal::Ciphertext t35 = snni_boot_ct(context, fenc, fencr, fev, scale, 35);
        std::vector<seal::Ciphertext> thread_ws;
        thread_ws.reserve(13);
        thread_ws.push_back(t35); snni_boot_touch(thread_ws.back());
        for (int i = 0; i < 12; ++i) { thread_ws.push_back(t32); snni_boot_touch(thread_ws.back()); }
        fprintf(stderr, "FLOOR_BOOT|thread_ws|count=%zu|limbs=%zu+12x%zu\n", thread_ws.size(),
                thread_ws[0].coeff_modulus_size(), thread_ws[1].coeff_modulus_size());
        fflush(stderr);
        snni_boot("boot_instant");
        long ctxp  = snni_boot_get("ctx_pub");
        long total = snni_boot_get("boot_instant");
        long relin = snni_boot_get("boot_context") - ctxp;
        long keys  = snni_boot_get("boot_after_stage") - snni_boot_get("boot_context");
        long chnk  = snni_boot_get("boot_after_chunk") - snni_boot_get("boot_after_stage");
        long thr   = total - snni_boot_get("boot_after_chunk");
        fprintf(stderr,
            "FLOOR_BOOT_SUMMARY|ctx_pub_kb=%ld|relin_kb=%ld|keys15_kb=%ld|chunk_kb=%ld|thread_kb=%ld|instant_kb=%ld|target_kb=%ld\n",
            ctxp, relin, keys, chnk, thr, total, total - ctxp);
        fflush(stderr);
        // extra, outside the target: one key switch at T=1 on a chunk ciphertext
        fev.rotate_vector_inplace(chunk[0], 1, stage_keys);
        snni_boot_touch(chunk[0]);
        snni_boot("boot_after_keyswitch");
        fprintf(stderr, "FLOOR_BOOT|keyswitch_step_kb=%ld\n",
                snni_boot_get("boot_after_keyswitch") - total);
        fflush(stderr);
        return;   // nothing else is allocated
    }""",
    "bootstrap floor-bench body",
    already='snni_boot("boot_context")',
)

print("\n--- bootstrap floor-bench sites now in the source ---")
hits = subprocess.run(
    ["grep", "-n", "snni_boot\\|FLOOR_BOOT", str(HDR)],
    capture_output=True, text=True).stdout.strip()
print(hits if hits else "(none -- the patch did not take)")
_txt = HDR.read_text()
ok = bool(hits) and 'snni_boot("boot_context")' in _txt and 'snni_boot("boot_instant")' in _txt and 'thread_ws' in _txt
print("PATCH_OK" if ok else "PATCH_FAIL")
sys.exit(0 if ok else 3)
