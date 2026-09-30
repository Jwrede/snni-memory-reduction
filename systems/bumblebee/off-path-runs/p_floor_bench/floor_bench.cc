// BumbleBee target micro-benchmark (thesis Appendix B.3, Section 4.5.5).
// Builds the two terms of the bumblebee target with SPU's own objects at the pinned commit and
// reads the resident footprint of each from /proc/self/status (VmRSS):
//
//   --mode ot --rank r   one OT instance, BasicOTProtocols(conn, YACL_Ferret): a Ferret sender and
//                        a Ferret receiver adapter, each with a buffer of lpn_param_.n x 16 B
//                        (n = 10,485,760). Measured after construction (OneTimeSetup) and after
//                        1,000,000 OT requests per direction. The first requests are served from
//                        the OneTimeSetup reserve (470,016 entries); exhausting it runs the Ferret
//                        bootstrap, which writes the whole buffer. Two processes, one per party,
//                        over a loopback brpc link.
//   --mode he            the target's HE term for the feed-forward up-projection
//                        (128 x 768) x (768 x 3072) at CheetahDot's FM64 parameters
//                        (N = 8192, coeff modulus {59, 55, 49, 55}, special prime, 3 carried primes):
//                        the encrypted activation from MatMatProtocol (BumbleBee's own sub-shape and
//                        encryption), the weight matrix at dense packing (768 x 3072 / 8192 = 288
//                        plaintexts in NTT form) and the response after packing (CeilDiv of
//                        MatMatProtocol's output count by its sub-shape width, 3072 / 64 = 48
//                        ciphertexts).
//   --mode he_stock      the same operation as the stock CheetahDot receiver holds it: the whole
//                        weight side encoded at MatMatProtocol's sub-shape and the unpacked result
//                        of Compute (the encoding blow-up step b1 removes).
//
// After each term malloc_trim(0) returns glibc's freed pages, so a delta counts live objects.

#include <malloc.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "seal/seal.h"
#include "yacl/link/link.h"

#include "libspu/mpc/cheetah/arith/matmat_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"

using namespace spu;
using namespace spu::mpc;
using namespace spu::mpc::cheetah;

static long rss_kb() {
  FILE* fp = fopen("/proc/self/status", "r");
  if (!fp) return -1;
  char line[256];
  long kb = -1;
  while (fgets(line, sizeof line, fp)) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      kb = strtol(line + 6, nullptr, 10);
      break;
    }
  }
  fclose(fp);
  return kb;
}

static long g_prev = 0;

static void mark(const char* who, const char* tag, long derived_b) {
  malloc_trim(0);
  long now = rss_kb();
  printf("FLOOR|%s|%s|derived_bytes=%ld|rss_delta_kb=%ld|vmrss_kb=%ld\n", who, tag,
         derived_b, now - g_prev, now);
  fflush(stdout);
  g_prev = now;
}

static int run_ot(int rank) {
  yacl::link::ContextDesc desc;
  desc.parties.push_back({"party0", "127.0.0.1:9630"});
  desc.parties.push_back({"party1", "127.0.0.1:9631"});
  auto lctx = yacl::link::FactoryBrpc().CreateContext(desc, rank);
  lctx->ConnectToMesh();
  auto comm = std::make_shared<Communicator>(lctx);
  const char* who = rank == 0 ? "ot_rank0" : "ot_rank1";

  const size_t n = 1000000;
  std::vector<uint64_t> o0(n, 1), o1(n, 1), r(n, 1);
  std::vector<uint8_t> ch(n, 1);

  malloc_trim(0);
  g_prev = rss_kb();
  const long base = g_prev;
  mark(who, "baseline", 0);

  auto prot = std::make_shared<BasicOTProtocols>(comm, CheetahOtKind::YACL_Ferret);
  mark(who, "instance_constructed", 0);
  if (rank == 0) {
    prot->GetSenderCOT()->SendRMRC(absl::MakeSpan(o0), absl::MakeSpan(o1), 64);
    prot->GetSenderCOT()->Flush();
    prot->GetReceiverCOT()->RecvRMRC(absl::MakeSpan(ch), absl::MakeSpan(r), 64);
  } else {
    prot->GetReceiverCOT()->RecvRMRC(absl::MakeSpan(ch), absl::MakeSpan(r), 64);
    prot->GetSenderCOT()->SendRMRC(absl::MakeSpan(o0), absl::MakeSpan(o1), 64);
    prot->GetSenderCOT()->Flush();
  }
  const long ot_derived = 2L * 10485760L * 16L;
  mark(who, "after_bootstrap", ot_derived);
  printf("FLOOR_SUMMARY|%s|derived_bytes=%ld|rss_delta_kb=%ld|rss_delta_bytes=%ld|ratio=%.4f\n",
         who, ot_derived, g_prev - base, (g_prev - base) * 1024,
         (double)(g_prev - base) * 1024.0 / (double)ot_derived);
  fflush(stdout);
  lctx->WaitLinkTaskFinish();
  return 0;
}

static int run_he(bool stock) {
  // CheetahDot::Impl::DecideSEALParameters(64) and LazyInit, verbatim parameters.
  seal::EncryptionParameters parms(seal::scheme_type::ckks);
  parms.set_poly_modulus_degree(8192);
  parms.set_use_special_prime(true);
  parms.set_coeff_modulus(seal::CoeffModulus::Create(8192, {59, 55, 49, 55}));
  seal::SEALContext context(parms, true, seal::sec_level_type::none);
  seal::KeyGenerator keygen(context);
  seal::SecretKey sk = keygen.secret_key();
  auto modulus = context.first_context_data()->parms().coeff_modulus();
  parms.set_coeff_modulus(modulus);
  seal::SEALContext ecd_ms_context(parms, false, seal::sec_level_type::none);
  ModulusSwitchHelper ecd_msh(ecd_ms_context, 64);
  MatMatProtocol prot(context, ecd_msh, /*disable_pack*/ false);
  const char* who = stock ? "he_stock" : "he";

  MatMatProtocol::Meta meta;
  meta.dims = {128, 768, 3072};
  auto sub = prot.GetSubMatShape(meta);
  const size_t ln = prot.GetLeftSize(meta), rn = prot.GetRightSize(meta),
               on = prot.GetOutSize(meta);
  const bool enc_lhs = ln <= rn;
  const size_t nmod = modulus.size();
  const size_t N = 8192;
  const long ct_b = 2L * N * (long)nmod * 8L, pt_b = (long)N * (long)nmod * 8L;
  const size_t rn_dense = (768 * 3072 + N - 1) / N;
  const size_t on_packed = (on + sub[1] - 1) / sub[1];
  printf("HE|subshape=%ldx%ldx%ld|lhs_polys=%zu|rhs_polys=%zu|out_polys=%zu|encrypt_lhs=%d"
         "|carried_primes=%zu|ct_bytes=%ld|pt_bytes=%ld|rhs_dense=%zu|out_packed=%zu\n",
         (long)sub[0], (long)sub[1], (long)sub[2], ln, rn, on, (int)enc_lhs, nmod, ct_b, pt_b,
         rn_dense, on_packed);
  if (!enc_lhs) {
    printf("HE|unexpected: CheetahDot would encrypt the right operand\n");
    return 1;
  }

  auto lhs = ring_rand(FieldType::FM64, {128, 768});
  auto rhs = ring_rand(FieldType::FM64, {768, 3072});
  seal::Encryptor encryptor(context, sk);
  auto ct_bytes = [](const std::vector<RLWECt>& v) {
    long b = 0;
    for (auto& c : v) b += (long)(c.size() * c.poly_modulus_degree() * c.coeff_modulus_size() * 8);
    return b;
  };
  auto pt_bytes = [](const std::vector<RLWEPt>& v) {
    long b = 0;
    for (auto& p : v) b += (long)(p.coeff_count() * 8);
    return b;
  };

  malloc_trim(0);
  g_prev = rss_kb();
  const long base = g_prev;
  mark(who, "baseline", 0);

  std::vector<RLWECt> lhs_ct(ln);
  {
    std::vector<RLWEPt> lhs_pt(ln);
    prot.EncodeLHS(lhs, meta, true, absl::MakeSpan(lhs_pt));
    for (size_t i = 0; i < ln; ++i) {
      NttInplace(lhs_pt[i], context);
      encryptor.encrypt_symmetric(lhs_pt[i], lhs_ct[i]);
    }
  }
  printf("HE|encrypted_activation|objects=%zu|object_bytes=%ld\n", ln, ct_bytes(lhs_ct));
  mark(who, "encrypted_activation", (long)ln * ct_b);

  long derived = (long)ln * ct_b;
  if (stock) {
    std::vector<RLWEPt> rhs_pt(rn);
    prot.EncodeRHS(rhs, meta, false, absl::MakeSpan(rhs_pt));
    for (auto& p : rhs_pt) NttInplace(p, context);
    printf("HE|plaintext_weights_stock|objects=%zu|object_bytes=%ld\n", rn, pt_bytes(rhs_pt));
    mark(who, "plaintext_weights_stock", (long)rn * pt_b);
    std::vector<RLWECt> out_ct(on);
    prot.Compute(absl::MakeSpan(lhs_ct), absl::MakeSpan(rhs_pt), meta, absl::MakeSpan(out_ct));
    printf("HE|result_unpacked|objects=%zu|object_bytes=%ld\n", on, ct_bytes(out_ct));
    mark(who, "result_unpacked", (long)on * ct_b);
    derived += (long)rn * pt_b + (long)on * ct_b;
    printf("FLOOR_SUMMARY|%s|derived_bytes=%ld|rss_delta_kb=%ld|rss_delta_bytes=%ld|ratio=%.4f\n",
           who, derived, g_prev - base, (g_prev - base) * 1024,
           (double)(g_prev - base) * 1024.0 / (double)derived);
    return 0;
  }

  // Weight matrix at dense packing: rn_dense plaintexts at the 3 carried primes, NTT form, every
  // coefficient a reduced residue (the encoded weight values; written so every page is resident).
  std::vector<RLWEPt> w_pt(rn_dense);
  {
    std::mt19937_64 rng(100);
    for (auto& p : w_pt) {
      p.parms_id() = seal::parms_id_zero;
      p.resize(N * nmod);
      for (size_t j = 0; j < nmod; ++j) {
        const uint64_t q = modulus[j].value();
        uint64_t* d = p.data() + j * N;
        for (size_t k = 0; k < N; ++k) d[k] = rng() % q;
      }
      p.parms_id() = context.first_parms_id();
      p.scale() = 1.0;
    }
  }
  printf("HE|plaintext_weights_dense|objects=%zu|object_bytes=%ld\n", rn_dense, pt_bytes(w_pt));
  mark(who, "plaintext_weights_dense", (long)rn_dense * pt_b);

  // Response after packing: on_packed ciphertexts at the 3 carried primes.
  std::vector<RLWECt> resp(on_packed);
  {
    RLWEPt zero;
    zero.parms_id() = seal::parms_id_zero;
    zero.resize(N * nmod);
    zero.parms_id() = context.first_parms_id();
    zero.scale() = 1.0;
    for (auto& c : resp) encryptor.encrypt_symmetric(zero, c);
  }
  printf("HE|result_packed|objects=%zu|object_bytes=%ld\n", on_packed, ct_bytes(resp));
  mark(who, "result_packed", (long)on_packed * ct_b);

  derived += (long)rn_dense * pt_b + (long)on_packed * ct_b;
  printf("FLOOR_SUMMARY|%s|derived_bytes=%ld|rss_delta_kb=%ld|rss_delta_bytes=%ld|ratio=%.4f\n",
         who, derived, g_prev - base, (g_prev - base) * 1024,
         (double)(g_prev - base) * 1024.0 / (double)derived);
  fflush(stdout);
  return 0;
}

int main(int argc, char** argv) {
  std::string mode = "he";
  int rank = 0;
  for (int i = 1; i + 1 < argc; ++i) {
    if (strcmp(argv[i], "--mode") == 0) mode = argv[i + 1];
    if (strcmp(argv[i], "--rank") == 0) rank = atoi(argv[i + 1]);
  }
  if (mode == "ot") return run_ot(rank);
  return run_he(mode == "he_stock");
}
