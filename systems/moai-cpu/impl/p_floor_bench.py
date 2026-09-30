#!/usr/bin/env python3
"""Probe-only target micro-benchmark: allocates only the protocol-required set at the bootstrap peak
at T=1 and reports resident footprint (VmRSS/VmHWM from /proc/self/status) per phase.

Difference from p_keyaudit.py: that reports serialised key size (GaloisKeys::save_size, lower bound);
this reports resident bytes (NTT form plus pool retention).
Reuses the live objects of all_layer_test (SEALContext, KeyGenerator, relin_keys, gal_keys (30),
gal_steps_vector, gal_keys_boot (47)): context and keys identical to the measured run. Guarded reads at
each construction site; bench block appended after the bootstrap keys and LT coefficients; returns
before the 12-layer inference.

    p_floor_bench.py [/root/moai]        applied to a fresh pinned checkout, probe images only,
                                         the same slot pin_seed_and_gate.py / p_keyaudit.py occupy
                                         (NOT stacked on p_keyaudit.py; both edit the same anchors)

Run with SNNI_FLOOR_BENCH=1 (and OMP_NUM_THREADS=1); also calls omp_set_num_threads(1).
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


# ------------------------------------------------------------------------------------------------
# 1. The floor-bench helpers, placed just before all_layer_test at global scope. Every FLOOR line
#    goes to stderr so it never touches the gate's stdout. Reads are no-ops unless SNNI_FLOOR_BENCH.
edit(
    HDR,
    "void all_layer_test(){\n    cout <<\"Task: test BERT in CKKS scheme: \"<<endl;",
    r"""// --- probe only: the floor micro-benchmark ---------------------------------------------
// Reports the RESIDENT footprint (VmRSS) of exactly the protocol-required set at the bootstrap peak,
// phase by phase, from /proc/self/status rather than from a size derivation. VmHWM (the transient
// high-water, which includes key-generation scratch) is printed beside it. All markers carry the
// literal prefixes FLOOR| and FLOOR_SUMMARY| so the build sbatch can confirm the patch took with
// `strings`.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <exception>
#ifdef _OPENMP
#include <omp.h>
#endif

static int snni_floor_enabled() {
    static int v = -1;
    if (v < 0) { const char *e = getenv("SNNI_FLOOR_BENCH"); v = (e && *e) ? 1 : 0; }
    return v;
}
static std::map<std::string, long> &snni_floor_map() { static std::map<std::string, long> m; return m; }
static void snni_read_status(long &rss_kb, long &hwm_kb) {
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
static void snni_floor(const char *name) {
    if (!snni_floor_enabled()) return;
    long rss = -1, hwm = -1;
    snni_read_status(rss, hwm);
    snni_floor_map()[std::string(name)] = rss;
    fprintf(stderr, "FLOOR|%s|vmrss_kb=%ld|vmhwm_kb=%ld\n", name, rss, hwm);
    fflush(stderr);
}
static long snni_floor_get(const char *name) {
    std::map<std::string, long> &m = snni_floor_map();
    std::map<std::string, long>::iterator it = m.find(std::string(name));
    return (it == m.end()) ? 0 : it->second;
}
static void snni_floor_begin() {
    if (!snni_floor_enabled()) return;
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    fprintf(stderr, "FLOOR|begin|note=T=1 protocol-only resident floor; deltas are VmRSS\n");
    fflush(stderr);
    snni_floor("after_start");
}
// Fault every backing page of a ciphertext so its cost is resident, not a lazy zero mapping.
static void snni_touch(const seal::Ciphertext &ct) {
    const uint64_t *p = ct.data();
    size_t n = (size_t)ct.size() * (size_t)ct.poly_modulus_degree() * (size_t)ct.coeff_modulus_size();
    volatile uint64_t s = 0;
    for (size_t i = 0; i < n; ++i) s ^= p[i];
    (void)s;
}
// Encrypt zero at the top level, mod-switch down to target_limbs RNS limbs, and touch it resident.
// remaining_level 20 -> 21 limbs (the dependency-live hidden state); 35 -> a modraised in-flight CT;
// 32 -> a Paterson-Stockmeyer temporary at the modular-reduction stage.
static seal::Ciphertext snni_make_ct(seal::SEALContext &context, seal::CKKSEncoder &encoder,
        seal::Encryptor &encryptor, seal::Evaluator &evaluator, double scale, size_t target_limbs) {
    (void)context;
    seal::Plaintext pt;
    encoder.encode(0.0, scale, pt);
    seal::Ciphertext ct;
    encryptor.encrypt(pt, ct);
    while (ct.coeff_modulus_size() > target_limbs) evaluator.mod_switch_to_next_inplace(ct);
    snni_touch(ct);
    return ct;
}
static void snni_floor_summary() {
    if (!snni_floor_enabled()) return;
    long ctx      = snni_floor_get("after_context");
    long relin    = snni_floor_get("after_relin")   - ctx;                       // + public_key (~0.04 GB)
    long gal30    = snni_floor_get("after_gal_keys") - snni_floor_get("after_relin");
    long boot     = snni_floor_get("after_gal_keys_boot") - snni_floor_get("before_gal_keys_boot");
    long per_key  = boot / 47;
    long stage15  = per_key * 15;                                                // largest BSGS stage + conj
    long stage_dir = snni_floor_get("after_largest_stage_keys") - snni_floor_get("before_largest_stage_keys");
    long ctvec    = snni_floor_get("after_2_ct_vectors") - snni_floor_get("before_ct_vectors");
    long inflight = snni_floor_get("after_inflight")     - snni_floor_get("after_2_ct_vectors");
    long target   = stage15 + relin + ctvec + inflight;                          // context excluded (overhead)
    fprintf(stderr, "FLOOR|gal_keys_30_bank_kb=%ld|largest_stage_direct_kb=%ld\n", gal30, stage_dir);
    fprintf(stderr,
        "FLOOR_SUMMARY|context_kb=%ld|relin_kb=%ld|boot_bank_kb=%ld|per_boot_key_kb=%ld"
        "|largest_stage15_kb=%ld|ct_vectors_kb=%ld|inflight_kb=%ld|target_kb=%ld\n",
        ctx, relin, boot, per_key, stage15, ctvec, inflight, target);
    fflush(stderr);
}
// ---------------------------------------------------------------------------------------

void all_layer_test(){
    cout <<"Task: test BERT in CKKS scheme: "<<endl;
    snni_floor_begin();""",
    "floor-bench helpers",
    already="FLOOR|%s|vmrss_kb=",
)

# ------------------------------------------------------------------------------------------------
# 2. after_context: the SEALContext is built (evaluation params + modulus chain), no keys yet.
edit(
    HDR,
    "    SEALContext context(parms, true, sec_level_type::none);",
    "    SEALContext context(parms, true, sec_level_type::none);\n    snni_floor(\"after_context\");",
    "after_context read",
    already='snni_floor("after_context")',
)

# ------------------------------------------------------------------------------------------------
# 3. after_relin: one relinearisation key resident (delta also carries the small public key above it).
edit(
    HDR,
    "    keygen.create_relin_keys(relin_keys);",
    "    keygen.create_relin_keys(relin_keys);\n    snni_floor(\"after_relin\");",
    "after_relin read",
    already='snni_floor("after_relin")',
)

# ------------------------------------------------------------------------------------------------
# 4. after_gal_keys: the 30-key layer bank (create_galois_keys, no argument). Optional, not in target.
edit(
    HDR,
    "    keygen.create_galois_keys(gal_keys);",
    "    keygen.create_galois_keys(gal_keys);\n    snni_floor(\"after_gal_keys\");",
    "after_gal_keys read",
    already='snni_floor("after_gal_keys")',
)

# ------------------------------------------------------------------------------------------------
# 5. before/after the 47-key bootstrap bank, so its resident cost is isolated from the LT
#    coefficients and the bootstrapper's mod-polynomial built around it.
edit(
    HDR,
    "    keygen.create_galois_keys(gal_steps_vector, gal_keys_boot);",
    "    snni_floor(\"before_gal_keys_boot\");\n"
    "    keygen.create_galois_keys(gal_steps_vector, gal_keys_boot);\n"
    "    snni_floor(\"after_gal_keys_boot\");",
    "around the bootstrap galois key set",
    already='snni_floor("before_gal_keys_boot")',
)

# ------------------------------------------------------------------------------------------------
# 6. The bench body, appended after the LT coefficients are generated (context, keygen, encoder,
#    encryptor, evaluator, relin_keys, gal_keys_boot, gal_steps_vector, scale are all in scope here).
#    Optionally measures the largest BSGS stage directly (14 keys of the basicstep-32 stage, P1),
#    then the dependency-live hidden state (2 x 768 CTs at 21 limbs) and one thread's in-flight
#    working set (a 35-limb modraised CT + 12 temporaries at 32 limbs), then the summary, then
#    returns before the inference so nothing else is allocated.
edit(
    HDR,
    "  bootstrapper.generate_LT_coefficient_3();",
    r"""  bootstrapper.generate_LT_coefficient_3();

    if (snni_floor_enabled()) {
        // (optional) direct residency of the largest BSGS stage: the 14 basicstep-32 rotations (P1),
        // which are literal entries of gal_steps_vector for logn=15. Guarded: if any is absent (a
        // different logn), skip rather than mismeasure; if generation throws, skip rather than abort.
        {
            std::vector<int> stage;
            stage.push_back(32);    stage.push_back(64);    stage.push_back(96);    stage.push_back(224);
            stage.push_back(448);   stage.push_back(672);   stage.push_back(896);   stage.push_back(31872);
            stage.push_back(32096); stage.push_back(32320); stage.push_back(32544); stage.push_back(32672);
            stage.push_back(32704); stage.push_back(32736);
            bool ok = true;
            for (size_t i = 0; i < stage.size(); ++i)
                if (std::find(gal_steps_vector.begin(), gal_steps_vector.end(), stage[i]) == gal_steps_vector.end()) { ok = false; break; }
            if (ok) {
                try {
                    snni_floor("before_largest_stage_keys");
                    seal::GaloisKeys stage_keys;
                    keygen.create_galois_keys(stage, stage_keys);
                    snni_floor("after_largest_stage_keys");
                } catch (const std::exception &e) {
                    fprintf(stderr, "FLOOR|after_largest_stage_keys|status=SKIPPED|why=%s\n", e.what());
                    fflush(stderr);
                }
            } else {
                fprintf(stderr, "FLOOR|after_largest_stage_keys|status=SKIPPED|why=steps_absent_logn_not_15\n");
                fflush(stderr);
            }
        }
        // dependency-live hidden state: two 768-wide vectors at remaining_level 20 (21 limbs).
        snni_floor("before_ct_vectors");
        std::vector<seal::Ciphertext> hs1, hs2, inflight;
        hs1.reserve(num_col); hs2.reserve(num_col);
        for (int i = 0; i < num_col; ++i) hs1.push_back(snni_make_ct(context, encoder, encryptor, evaluator, scale, 21));
        for (int i = 0; i < num_col; ++i) hs2.push_back(snni_make_ct(context, encoder, encryptor, evaluator, scale, 21));
        snni_floor("after_2_ct_vectors");
        // one thread's in-flight working set: a modraised CT at 35 limbs plus the Paterson-Stockmeyer
        // temporaries (baby[8] + giant[3] + tmp) at ~32 limbs.
        inflight.push_back(snni_make_ct(context, encoder, encryptor, evaluator, scale, 35));
        for (int i = 0; i < 12; ++i) inflight.push_back(snni_make_ct(context, encoder, encryptor, evaluator, scale, 32));
        snni_floor("after_inflight");
        snni_floor_summary();
        // hs1, hs2, inflight stay resident through the reads above; released at return.
        return;
    }""",
    "floor-bench body",
    already='snni_floor("before_ct_vectors")',
)

# ------------------------------------------------------------------------------------------------
print("\n--- every floor-bench site now in the source ---")
hits = subprocess.run(
    ["grep", "-n", "snni_floor\\|snni_make_ct\\|FLOOR_SUMMARY\\|snni_touch", str(HDR)],
    capture_output=True, text=True).stdout.strip()
print(hits if hits else "(none -- the patch did not take)")
print("--- end ---\n")

_txt = HDR.read_text()
ok = bool(hits) and 'snni_floor("before_ct_vectors")' in _txt and 'snni_floor("after_inflight")' in _txt
print("PATCH_OK" if ok else "PATCH_FAIL")
sys.exit(0 if ok else 3)
