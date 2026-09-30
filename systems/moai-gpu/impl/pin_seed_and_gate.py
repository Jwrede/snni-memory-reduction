#!/usr/bin/env python3
"""Pins the seed and emits a gate observable (pin_seed_and_gate.py <source-root>, in place).

Pinned from SNNI_SEED: random_bytes in prng.cuh (key material) and adjust_sk_hamming_weight in
secretkey.cu:376 (surviving secret-key coefficients). Callers serial (no omp/threads in phantom src);
mutex added anyway. Gate: full-double fingerprint (count/sum/sumsq/absmax + first eight values) over
the 768 output ciphertexts, after the last bootstrap (live set far below the peak). Degeneracy guard:
fail if all values are zero. Compared against the baseline only (output not claimed correct;
MANIFEST).
"""
import sys

PRNG_OLD = """void inline random_bytes(unsigned char *buf, size_t count, const cudaStream_t &stream) {
    std::random_device rd;"""

PRNG_NEW = r"""// SNNI campaign: determinism. Every random draw in this stack passes through here, and all of
// its callers are serial (verified: no omp, no threads in phantom's src). See
// systems/moai-gpu/impl/pin_seed_and_gate.py. The seed is part of the measured system and is
// declared in the MANIFEST; it is NOT a deployment configuration, and fixed key material would be
// a security defect in one.
void inline random_bytes(unsigned char *buf, size_t count, const cudaStream_t &stream) {
    static std::mutex snni_prng_mx;
    static std::mt19937_64 snni_prng = [] {
        const char *e = std::getenv("SNNI_SEED");
        unsigned long long s = e ? std::strtoull(e, nullptr, 0) : 0x5EEDULL;
        std::fprintf(stderr, "SEEDED|phantom_random_bytes|seed=0x%llx\n", s);
        std::fflush(stderr);
        return std::mt19937_64(s);
    }();
    std::random_device rd;
    (void)rd;"""

PRNG_DIST_OLD = "        i = dist(rd);"
PRNG_DIST_NEW = """        std::lock_guard<std::mutex> snni_lk(snni_prng_mx);
        i = dist(snni_prng);"""

INCLUDES_ANCHOR = "#include <random>"
INCLUDES_NEW = """#include <random>
// SNNI campaign additions, for the pinned generator below.
#include <cstdio>
#include <cstdlib>
#include <mutex>"""

GATE_BLOCK = r"""
    // ---- SNNI campaign: the gate observable -------------------------------------------------
    // Compared against the BASELINE's, never against an external reference: that is what makes it
    // a test of whether THIS STEP changed anything. Emitted here, after the last bootstrap, where
    // the live set is far below the peak, so it cannot move the number being published.
    {
        double g_sum = 0.0, g_sumsq = 0.0, g_absmax = 0.0;
        long long g_n = 0, g_nonzero = 0;
        std::cout.precision(17);
        for (int i = 0; i < (int)rtn2.size(); ++i) {
            PhantomPlaintext g_pt;
            decryptor.decrypt(rtn2[i], g_pt);
            vector<double> g_v;
            encoder.decode(g_pt, g_v);
            for (int ind = 0; ind < (int)slot_count; ++ind) {
                if (b_vec[ind] != 1) continue;
                double x = g_v[ind];
                g_sum += x; g_sumsq += x * x; ++g_n;
                if (x != 0.0) ++g_nonzero;
                double ax = x < 0 ? -x : x;
                if (ax > g_absmax) g_absmax = ax;
            }
            if (i == 0) {
                int shown = 0;
                for (int ind = 0; ind < (int)slot_count && shown < 8; ++ind) {
                    if (b_vec[ind] != 1) continue;
                    std::cout << "GATE|ct0[" << ind << "]=" << g_v[ind] << std::endl;
                    ++shown;
                }
            }
        }
        std::cout << "GATE|out|n=" << g_n << "|sum=" << g_sum << "|sumsq=" << g_sumsq
                  << "|absmax=" << g_absmax << std::endl;
        std::cout << "GATE_NONZERO|" << g_nonzero << "/" << g_n << std::endl;
        // A gate that cannot fail is worse than none: it reads as evidence while proving nothing.
        // SHARK's first attempt reproduced perfectly over 98,304 exact zeros, and SIGMA-GPU's
        // argmax was trivially 0 under input.zero(). So refuse a constant-zero observable here
        // rather than discover it after a line has been published against it.
        if (g_nonzero == 0) {
            std::cout << "GATE_DEGENERATE|every observable value is zero" << std::endl;
            std::cerr << "FAILED: gate observable is all zeros, it cannot distinguish anything"
                      << std::endl;
            std::exit(6);
        }
    }
    // -----------------------------------------------------------------------------------------
"""


SK_OLD = """    // Random device and generator for shuffling and random index generation
    std::random_device rd;
    std::mt19937 gen(rd());"""

SK_NEW = r"""    // SNNI campaign: determinism, SECOND draw source. `prng.cuh` was pinned first and was not
    // enough: this function decides WHICH coefficients of the secret key survive the hamming-weight
    // adjustment, by shuffling with its own unseeded generator. Pinning the values while leaving
    // the positions random still gives a different secret key on every run, which is exactly what
    // the two-run check measured on 2026-08-02 (jobs 45633998/99: per-slot output values
    // uncorrelated, aggregate statistics reproducible to about 3%). The claim that `random_bytes`
    // is the single source of every draw in this stack was wrong, and it was wrong in the one place
    // that decides the key.
    static std::mutex snni_sk_mx;
    static std::mt19937 gen = [] {
        const char *e = std::getenv("SNNI_SEED");
        unsigned long long s = e ? std::strtoull(e, nullptr, 0) : 0x5EEDULL;
        std::fprintf(stderr, "SEEDED|phantom_sk_hamming|seed=0x%llx\n", s);
        std::fflush(stderr);
        return std::mt19937(static_cast<unsigned int>(s ^ (s >> 32)));
    }();
    std::lock_guard<std::mutex> snni_sk_lk(snni_sk_mx);
    std::random_device rd;
    (void)rd;"""

SK_INCLUDES_ANCHOR = "#include <random>"
SK_INCLUDES_NEW = """#include <random>
// SNNI campaign additions, for the pinned generator below.
#include <cstdio>
#include <cstdlib>
#include <mutex>"""


def fail(msg):
    sys.exit(f"pin_seed_and_gate.py: {msg}")


def patch_prng(path):
    src = open(path).read()
    if "snni_prng" in src:
        print("  prng.cuh: already pinned, skipping")
        return False
    if PRNG_OLD not in src:
        fail(f"{path} does not contain the expected random_bytes body; upstream changed")
    if INCLUDES_ANCHOR not in src:
        fail(f"{path} has no '{INCLUDES_ANCHOR}' to hang the added includes on")
    src = src.replace(INCLUDES_ANCHOR, INCLUDES_NEW, 1)
    src = src.replace(PRNG_OLD, PRNG_NEW, 1)
    if PRNG_DIST_OLD not in src:
        fail(f"{path} does not contain the expected draw '{PRNG_DIST_OLD.strip()}'")
    src = src.replace(PRNG_DIST_OLD, PRNG_DIST_NEW, 1)
    open(path, "w").write(src)
    print("  prng.cuh: generator pinned, SEEDED| marker added, draw serialised")
    return True


def patch_gate(path):
    src = open(path).read()
    if "GATE|out|" in src:
        print("  test_single_layer.cuh: already emits a gate, skipping")
        return False
    anchor = "    snni_gpupeak(0);\n"
    if anchor not in src:
        fail(f"{path} has no 'snni_gpupeak(0);' -- run instrument.py first, so that the gate is "
             "emitted before the peak line rather than after it")
    src = src.replace(anchor, GATE_BLOCK + anchor, 1)
    open(path, "w").write(src)
    print("  test_single_layer.cuh: GATE| fingerprint and degeneracy guard inserted")
    return True


KEYFP_ANCHOR = '    snni_mem_marker("inputs_prepared");\n'

KEYFP_BLOCK = r"""    // ---- SNNI campaign: does the KEY MATERIAL reproduce? --------------------------------------
    // The two SEEDED| markers prove a generator was constructed with the seed. They do NOT prove
    // that every draw shaping the key runs through it, and that difference is the whole question
    // once two unchanged runs disagree on the output: either something is still unpinned, or the
    // randomness is fine and the divergence is in the arithmetic. Nothing published so far can tell
    // those apart, because the gate observable depends on both.
    //
    // A freshly encrypted input ciphertext depends on the plaintext (read from files, fixed), on
    // the key, and on the encryption randomness -- and on nothing that happens later. So its raw
    // limbs answer the question by themselves: identical across two runs means the key material and
    // the encryption randomness are pinned and any later divergence is arithmetic; different means
    // a draw is still open and the hunt continues. It is read before the attention block, where the
    // live set is far below the peak, and it is one 15.7 MB device-to-host copy.
    {
        const auto &g_ct = enc_ecd_x[0];
        size_t g_n = (size_t)g_ct.size() * (size_t)g_ct.coeff_modulus_size()
                     * (size_t)g_ct.poly_modulus_degree();
        std::vector<uint64_t> g_h(g_n);
        cudaMemcpy(g_h.data(), g_ct.data(), g_n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        unsigned long long g_fnv = 14695981039346656037ULL;
        for (size_t g_i = 0; g_i < g_n; ++g_i) {
            g_fnv ^= (unsigned long long)g_h[g_i];
            g_fnv *= 1099511628211ULL;
        }
        std::cout << "KEYFP|ct0|n=" << g_n << "|fnv=" << g_fnv << std::endl;
    }
    // -------------------------------------------------------------------------------------------
"""


def patch_keyfp(path):
    """Fingerprint one freshly encrypted ciphertext: identical across runs means key material and
    encryption randomness are pinned and any later divergence is arithmetic."""
    src = open(path).read()
    if "KEYFP|ct0|" in src:
        print("  test_single_layer.cuh: already emits a key fingerprint, skipping")
        return False
    if KEYFP_ANCHOR not in src:
        fail(f"{path} has no 'snni_mem_marker(\"inputs_prepared\");' to hang the fingerprint on -- "
             f"run instrument.py first")
    src = src.replace(KEYFP_ANCHOR, KEYFP_BLOCK + KEYFP_ANCHOR, 1)
    open(path, "w").write(src)
    print("  test_single_layer.cuh: KEYFP| ciphertext fingerprint inserted")
    return True


def patch_secretkey(path):
    """Pin the generator that shuffles which secret-key coefficients stay non-zero. Skipped (not
    refused) when already applied, so it can be added to a tree whose prng.cuh is already pinned."""
    src = open(path).read()
    if "snni_sk_mx" in src:
        print("  secretkey.cu: already pinned, skipping")
        return False
    if SK_OLD not in src:
        fail(f"{path} does not contain the expected hamming-weight generator; upstream changed")
    if SK_INCLUDES_ANCHOR not in src:
        fail(f"{path} has no '{SK_INCLUDES_ANCHOR}' to hang the added includes on")
    src = src.replace(SK_INCLUDES_ANCHOR, SK_INCLUDES_NEW, 1)
    src = src.replace(SK_OLD, SK_NEW, 1)
    open(path, "w").write(src)
    print("  secretkey.cu: hamming-weight shuffle pinned, SEEDED| marker added")
    return True


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = sys.argv[1].rstrip("/")
    patch_prng(f"{root}/thirdparty/phantom-fhe/include/prng.cuh")
    patch_secretkey(f"{root}/thirdparty/phantom-fhe/src/secretkey.cu")
    patch_gate(f"{root}/src/include/test/test_single_layer.cuh")
    patch_keyfp(f"{root}/src/include/test/test_single_layer.cuh")
    print("done. seed pinned, gate emitted, degeneracy refused.")


def patch_seed_only(root):
    """Add just the second draw source to a tree that already carries the rest."""
    return patch_secretkey(f"{root}/thirdparty/phantom-fhe/src/secretkey.cu")


def patch_keyfp_only(root):
    """Add just the key fingerprint to a tree that already carries the rest."""
    return patch_keyfp(f"{root}/src/include/test/test_single_layer.cuh")


if __name__ == "__main__":
    main()
