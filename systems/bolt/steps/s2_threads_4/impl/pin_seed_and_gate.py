#!/usr/bin/env python3
"""Pin BOLT's three randomness draw sources (PRG128, PRG256, SEAL PRNG) and add a gate observable,
in its own source tree. Run inside the image build on the freshly cloned EzPC/SCI tree, before cmake.
Each source is patched and announced separately; the build refuses a tree where any marker is missing.
The seed is part of the measured system (identical across baseline and steps), not a deployment config."""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/EzPC/SCI")


def edit(path, old, new, what, already):
    """Apply one patch idempotently; `already` is a required marker present ONLY post-patch (deriving
    it from `new` silently dropped six of eight patches in the first version)."""
    p = root / path
    src = p.read_text()
    if already in src:
        print(f"  already patched: {what}")
        return
    if old not in src:
        sys.exit(f"FAILED: anchor for '{what}' not found in {path}. "
                 f"Upstream moved; re-derive the patch rather than loosening the match.")
    p.write_text(src.replace(old, new, 1))
    src2 = p.read_text()
    if already not in src2:
        sys.exit(f"FAILED: applied '{what}' but its marker is still absent from {path}.")
    print(f"  patched: {what}  ({path})")


# --------------------------------------------------------------------------------------------
# 1. sci::PRG128, the OT and share randomness. Seed + per-instance construction counter keeps
# instances distinct (identical seeding would be a broken protocol, not determinism).
edit(
    "src/utils/prg.h",
    """      block128 v;
#ifdef EMP_USE_RANDOM_DEVICE""",
    """      block128 v;
      const char *snni_seed_env = getenv("SNNI_SEED");
      if (snni_seed_env != nullptr) {
        static std::atomic<uint64_t> snni_prg_counter(0);
        static std::atomic<int> snni_prg_announced(0);
        uint64_t snni_seed = strtoull(snni_seed_env, nullptr, 0);
        uint64_t n = snni_prg_counter.fetch_add(1);
        // splitmix64-style mixing so consecutive counters give unrelated keys.
        uint64_t z = snni_seed + (n + 1) * 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        uint64_t w = snni_seed ^ ((n + 1) * 0xD1B54A32D192ED03ULL);
        v = makeBlock128((long long)(z ^ (z >> 31)), (long long)w);
        if (snni_prg_announced.fetch_add(1) == 0) {
          fprintf(stderr, "SEEDED|prg128|seed=0x%llx\\n",
                  (unsigned long long)snni_seed);
          fflush(stderr);
        }
        reseed(&v);
        return;
      }
#ifdef EMP_USE_RANDOM_DEVICE""",
    "PRG128 default seeding",
    already="SEEDED|prg128|",
)

# --------------------------------------------------------------------------------------------
# 1b. sci::PRG256, the second draw source in this header, used by KKOT / split-KKOT OT in every
# non-linear layer. Pinning only PRG128 would print a SEEDED marker, pass every check, yet not reproduce.
edit(
    "src/utils/prg.h",
    """      alignas(32) block256 v;
#ifdef EMP_USE_RANDOM_DEVICE""",
    """      alignas(32) block256 v;
      const char *snni_seed_env = getenv("SNNI_SEED");
      if (snni_seed_env != nullptr) {
        static std::atomic<uint64_t> snni_prg256_counter(0);
        static std::atomic<int> snni_prg256_announced(0);
        uint64_t snni_seed = strtoull(snni_seed_env, nullptr, 0);
        uint64_t n = snni_prg256_counter.fetch_add(1);
        uint64_t *vw = (uint64_t *)(&v);
        for (int qi = 0; qi < 4; qi++) {
          uint64_t z = snni_seed + ((n * 4 + qi + 1) * 0x9E3779B97F4A7C15ULL);
          z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
          z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
          vw[qi] = z ^ (z >> 31);
        }
        if (snni_prg256_announced.fetch_add(1) == 0) {
          fprintf(stderr, "SEEDED|prg256|seed=0x%llx\\n",
                  (unsigned long long)snni_seed);
          fflush(stderr);
        }
        reseed(&v);
        return;
      }
#ifdef EMP_USE_RANDOM_DEVICE""",
    "PRG256 default seeding (KKOT / split-KKOT)",
    already="SEEDED|prg256|",
)

# The patch above uses atomic, getenv, strtoull and fprintf; prg.h includes none of them.
edit(
    "src/utils/prg.h",
    "namespace sci {",
    """#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace sci {""",
    "prg.h includes for the seeded path",
    already="#include <atomic>",
)

# --------------------------------------------------------------------------------------------
# 2. SEAL's PRNG, which decides the keys and every encryption's noise.
edit(
    "tests/bert_bolt/he.cpp",
    """parms.set_plain_modulus(plain_mod);""",
    """parms.set_plain_modulus(plain_mod);

	// SNNI: pin SEAL's randomness. Without this the secret key, the public key and every
	// encryption's noise come from a random_device seed, so no two runs produce the same
	// ciphertexts and no step can be compared against the baseline.
	{
		const char *snni_seed_env = getenv("SNNI_SEED");
		if (snni_seed_env != nullptr) {
			uint64_t snni_seed = strtoull(snni_seed_env, nullptr, 0);
			seal::prng_seed_type seal_seed{};
			for (size_t i = 0; i < seal_seed.size(); i++) {
				seal_seed[i] = snni_seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1);
			}
			parms.set_random_generator(
				std::make_shared<seal::Blake2xbPRNGFactory>(seal_seed));
			static int snni_seal_announced = 0;
			if (snni_seal_announced++ == 0) {
				fprintf(stderr, "SEEDED|seal_prng|seed=0x%llx\\n",
				        (unsigned long long)snni_seed);
				fflush(stderr);
			}
		}
	}""",
    "SEAL PRNG factory",
    already="SEEDED|seal_prng|",
)

edit(
    "tests/bert_bolt/he.cpp",
    """#include "he.h"
""",
    """#include "he.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
""",
    "he.cpp includes for the seeded path",
    already="#include <memory>",
)

# --------------------------------------------------------------------------------------------
# 3. The gate observable: BOB's two reconstructed class scores at full precision (%.17g of an
# integer / 2^NL_SCALE), to stderr and the gate file. ALICE holds no result, so the observable is BOB's.
edit(
    "tests/bert_bolt/bolt_bert.cpp",
    """            if(result.size() == 1){""",
    """            // SNNI GATE: the reconstructed class scores, at full precision, from the same
            // run whose peak is published. Emitted before the pretty-printed file below so a
            // format change to that file cannot silently change what the gate compares.
            {
                const char *gf = getenv("SNNI_GATE_FILE");
                FILE *gfp = (gf != nullptr) ? fopen(gf, "a") : nullptr;
                for(size_t gi = 0; gi < result.size(); gi++){
                    fprintf(stderr, "GATE|logit|sample=%d|i=%zu|%.17g\\n", i, gi, result[gi]);
                    if(gfp != nullptr) fprintf(gfp, "%.17g\\n", result[gi]);
                }
                fflush(stderr);
                if(gfp != nullptr){ fclose(gfp); }
                else if(gf != nullptr){ fprintf(stderr, "GATEFAIL|cannot open %s\\n", gf); }
            }
            if(result.size() == 1){""",
    "gate observable on BOB",
    already="SNNI GATE:",
)

# Both parties announce the workload they ran (a different sample/class count is a different program).
edit(
    "tests/bert_bolt/bolt_bert.cpp",
    """    cout << ">>> Evaluating Bert" << endl;""",
    """    fprintf(stderr, "WORKLOAD|party=%d|num_class=%d|sample_id=%d|num_sample=%d|bitlength=%d\\n",
            party, num_class, sample_id, num_sample, bitlength);
    fflush(stderr);
    cout << ">>> Evaluating Bert" << endl;""",
    "workload marker",
    already="WORKLOAD|party=",
)

edit(
    "tests/bert_bolt/bolt_bert.cpp",
    """#include "bert.h"
#include <fstream>""",
    """#include "bert.h"
#include <fstream>
#include <cstdio>
#include <cstdlib>""",
    "bolt_bert.cpp includes",
    already="#include <cstdio>",
)

# --------------------------------------------------------------------------------------------
# 4. Report any draw source not covered here (rand() without srand() is deterministic by accident).
import subprocess

hits = subprocess.run(
    ["grep", "-rn", "-E", r"\brand\(\)|srand\(|random_device|mt19937|chrono::.*now\(\)",
     "--include=*.cpp", "--include=*.h", str(root / "src"), str(root / "tests" / "bert_bolt")],
    capture_output=True, text=True).stdout.strip()
print("\n--- remaining draw sources and clock reads in the measured tree ---")
print(hits if hits else "(none)")
print("--- end ---\n")
print("PATCH_OK")
