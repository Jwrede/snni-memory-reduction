// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "libspu/mpc/cheetah/arith/cheetah_dot.h"

#include <algorithm>
#include <future>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "seal/batchencoder.h"
#include "seal/context.h"
#include "seal/decryptor.h"
#include "seal/encryptor.h"
#include "seal/evaluator.h"
#include "seal/galoiskeys.h"
#include "seal/keygenerator.h"
#include "seal/publickey.h"
#include "seal/secretkey.h"
#include "seal/util/locks.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "spdlog/spdlog.h"
#include "yacl/link/link.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/arith/matmat_prot.h"
#include "libspu/mpc/cheetah/rlwe/lwe_ct.h"
#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"
#include "libspu/mpc/cheetah/rlwe/packlwes.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

enum class CipherPackingType {
  lwes,
  rlwes,
  none,
};

static std::string ToString(CipherPackingType type) {
  switch (type) {
    case CipherPackingType::rlwes:
      return "interleave";
    case CipherPackingType::lwes:
      return "pack_lwes";
    default:
    case CipherPackingType::none:
      return "none";
  }
}

struct CheetahDot::Impl : public EnableCPRNG {
 public:
  const bool kUseModDownOptimization = true;
  static constexpr size_t kCtAsyncParallel = 16;

  explicit Impl(std::shared_ptr<yacl::link::Context> lctx,
                bool disable_matmul_pack)
      : lctx_(std::move(lctx)), disable_pack_(disable_matmul_pack) {}

  ~Impl() = default;

  // Compute C = A*B where |A|=dims[0]xdims[1], |B|=dims[1]xdims[2
  NdArrayRef DotOLE(const NdArrayRef &prv_mat, yacl::link::Context *conn,
                    const Shape3D &dim3, bool is_self_lhs);

  // dim4 = [B, M, K, L]
  // LHS.shape BxMxK, RHS.shape BxKxL
  // Out.shape BxMxL
  NdArrayRef BatchDotOLE(const NdArrayRef &prv_mat, yacl::link::Context *conn,
                         const Shape4D &dim4, bool is_self_lhs);

  NdArrayRef doDotOLE(const NdArrayRef &prv_mat, yacl::link::Context *conn,
                      const Shape3D &dim3, bool is_self_lhs);

  NdArrayRef doBatchDotOLE(const NdArrayRef &prv_mat, yacl::link::Context *conn,
                           const Shape4D &dim4, bool is_self_lhs);

  void doDotOLESenderSendStep(const NdArrayRef &prv_mat, const Shape3D &dim3,
                              bool is_self_lhs, CipherPackingType cptype,
                              yacl::link::Context *conn);

  // `num_packed` doubles as the permission to pack inside this call (campaign step
  // b2_result_ct_pack_free). A caller that passes nullptr gets the stock behaviour: the raw
  // result ciphertexts come back unpacked and the send step packs them. A caller that passes a
  // pointer allows the packing to be fused into the slice loop, and receives the number of packed
  // ciphertexts, which is what the send step would otherwise have computed for itself.
  void doDotOLEReceiverRecvStep(const NdArrayRef &prv_mat, const Shape3D &dim3,
                                bool is_self_lhs, CipherPackingType cptype,
                                absl::Span<RLWECt> result_cts,
                                yacl::link::Context *conn,
                                size_t *num_packed = nullptr);

  NdArrayRef doDotOLESenderRecvStep(FieldType field, size_t batch_size,
                                    MatMatProtocol::Meta meta,
                                    size_t num_ct_to_recv,
                                    CipherPackingType cptype,
                                    yacl::link::Context *conn);

  NdArrayRef doDotOLEReceiverSendStep(FieldType field, size_t batch_size,
                                      MatMatProtocol::Meta meta,
                                      absl::Span<RLWECt> ct_array_to_pack,
                                      CipherPackingType cptype,
                                      yacl::link::Context *conn,
                                      size_t bytes_recv,
                                      size_t pre_packed = 0);

  bool IsPackingEnabled(uint32_t ring_bitlen) const {
    return DecideSEALParameters(ring_bitlen).use_special_prime();
  }

  seal::EncryptionParameters DecideSEALParameters(uint32_t ring_bitlen) const {
    auto scheme_type = seal::scheme_type::ckks;
    auto parms = seal::EncryptionParameters(scheme_type);

    size_t poly_deg;
    std::vector<int> modulus_bits;
    // NOTE(lwj): we need Q=sum(modulus_bits) > 2*k for multiplying two
    // k-bit elements.
    // 1. We need the (N, Q) pair satisifies the security.
    //    Check `seal/util/globals.cpp` for the recommendation HE parameters.
    // 2. We prefer modulus_bits[i] to be around 49-bit aiming to use AVX512
    // for acceleration if avaiable.
    // 3. We need some bits for margin. That is Q > 2*k + margin for errorless
    // w.h.p. We set margin=32bit
    if (ring_bitlen <= 32) {
      poly_deg = 4096;
      // ~ 64 + 32 bit
      modulus_bits = {59, 37};
      parms.set_use_special_prime(false);
    } else if (ring_bitlen <= 64) {
      poly_deg = 8192;
      //  ~ 128 + 32 bit
      modulus_bits = {59, 55, 49, 55};
      parms.set_use_special_prime(true);
    } else {
      poly_deg = 16384;
      // ~ 256 + 30 bit
      modulus_bits = {59, 59, 59, 59, 49, 55};
      parms.set_use_special_prime(true);
    }

    parms.set_poly_modulus_degree(poly_deg);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_deg, modulus_bits));
    return parms;
  }

  void LazyInit(size_t field_bitlen, bool need_galois_key = false);

  void LazyInitGaloisKey(size_t field_bitlen);

  // Enc(Delta*m) -> Enc(Delta*m - r), r where r is sampled from Rq
  // One share of `m` is round(r/Delta) mod t
  // The other share of `m` is Dec(Enc(Delta*m - r))
  void H2A(absl::Span<RLWECt> ct, absl::Span<RLWEPt> rnd_mask,
           size_t target_modulus_size, const seal::PublicKey &pk,
           const seal::SEALContext &context) {
    seal::Evaluator evaluator(context);
    size_t num_poly = ct.size();
    SPU_ENFORCE(num_poly > 0);
    SPU_ENFORCE_EQ(rnd_mask.size(), num_poly);

    constexpr int64_t heuristic_group = 4;
    yacl::parallel_for(
        0, num_poly, heuristic_group, [&](size_t bgn, size_t end) {
          RLWECt zero_ct;
          for (size_t idx = bgn; idx < end; ++idx) {
            // NOTE(lwj): we hope the final ct is in the non-ntt form
            // We perform the intt before the modulus down which is faster
            // than modulus down then intt.
            InvNttInplace(ct[idx], context);

            ModulusSwtichInplace(ct[idx], target_modulus_size, context);

            // TODO(lwj): improve the performance of pk encryption of zero.
            // ct <- ct + enc(0)
            if (0 == zero_ct.size()) {
              seal::util::encrypt_zero_asymmetric(
                  pk, context, ct[idx].parms_id(), ct[idx].is_ntt_form(),
                  zero_ct);
            }

            evaluator.add_inplace(ct[idx], zero_ct);
            SPU_ENFORCE(!ct[idx].is_ntt_form());

            // sample r <- Rq
            // (ct[0] - r, ct[1]) <- ct
            UniformPoly(context, &rnd_mask[idx], ct[idx].parms_id());
            SubPlainInplace(ct[idx], rnd_mask[idx], context);
          }
        });
  }

 private:
  std::shared_ptr<yacl::link::Context> lctx_;
  bool disable_pack_ = false;

  // field_bitlen -> functor mapping
  std::unordered_map<size_t, std::shared_ptr<seal::SEALContext>> seal_cntxts_;
  std::unordered_map<size_t, std::shared_ptr<seal::SecretKey>> secret_keys_;
  std::unordered_map<size_t, std::shared_ptr<seal::PublicKey>> peer_pub_keys_;
  std::unordered_map<size_t, std::shared_ptr<seal::GaloisKeys>>
      peer_galois_keys_;
  // ModulusSwitchHelper for encoding
  std::unordered_map<size_t, std::shared_ptr<ModulusSwitchHelper>> ecd_mswh_;
  // ModulusSwitchHelper for decoding
  std::unordered_map<size_t, std::shared_ptr<ModulusSwitchHelper>> dcd_mswh_;
  std::unordered_map<size_t, std::shared_ptr<seal::Decryptor>> decryptors_;
};

void CheetahDot::Impl::LazyInitGaloisKey(size_t field_bitlen) {
  if (peer_galois_keys_.find(field_bitlen) != peer_galois_keys_.end()) {
    return;
  }
  auto kv = seal_cntxts_.find(field_bitlen);
  SPU_ENFORCE(kv != seal_cntxts_.end());
  const auto &this_context = *kv->second;
  const auto &this_rlwe_sk = *secret_keys_.find(field_bitlen)->second;

  seal::GaloisKeys gk;
  GenerateGaloisKeyForPacking(this_context, this_rlwe_sk,
                              /*seed*/ true, &gk);

  auto gk_buf = EncodeSEALObject(gk);
  int nxt_rank = lctx_->NextRank();
  std::shared_ptr<seal::GaloisKeys> peer_galois_key;
  if (nxt_rank == 0) {
    lctx_->Send(nxt_rank, gk_buf, "Rank0 send galois key");
    auto recv_gk = lctx_->Recv(nxt_rank, "Rank0 recv galois key");
    peer_galois_key = std::make_shared<seal::GaloisKeys>();
    DecodeSEALObject(recv_gk, this_context, peer_galois_key.get());
  } else {
    auto recv_gk = lctx_->Recv(nxt_rank, "Rank1 recv galois key");
    lctx_->Send(nxt_rank, gk_buf, "Rank0 send galois key");
    peer_galois_key = std::make_shared<seal::GaloisKeys>();
    DecodeSEALObject(recv_gk, this_context, peer_galois_key.get());
  }
  peer_galois_keys_.emplace(field_bitlen, peer_galois_key);
}

void CheetahDot::Impl::LazyInit(size_t field_bitlen, bool need_galois_keys) {
  if (seal_cntxts_.find(field_bitlen) != seal_cntxts_.end()) {
    if (need_galois_keys) {
      LazyInitGaloisKey(field_bitlen);
    }
    return;
  }

  auto parms = DecideSEALParameters(field_bitlen);
  auto *this_context =
      new seal::SEALContext(parms, true, seal::sec_level_type::none);
  seal::KeyGenerator keygen(*this_context);
  auto *rlwe_sk = new seal::SecretKey(keygen.secret_key());

  // exchange the public keys
  int nxt_rank = lctx_->NextRank();
  auto pk = keygen.create_public_key();
  auto pk_buf = EncodeSEALObject(pk.obj());
  auto peer_public_key = std::make_shared<seal::PublicKey>();
  if (nxt_rank == 0) {
    lctx_->Send(nxt_rank, pk_buf, "Rank1 send public key");
    auto recv_pk = lctx_->Recv(nxt_rank, "Rank1 recv public key");
    DecodeSEALObject(recv_pk, *this_context, peer_public_key.get());
  } else {
    auto recv_pk = lctx_->Recv(nxt_rank, "Rank0 recv public key");
    lctx_->Send(nxt_rank, pk_buf, "Rank1 send public key");
    DecodeSEALObject(recv_pk, *this_context, peer_public_key.get());
  }

  auto modulus = this_context->first_context_data()->parms().coeff_modulus();
  size_t enc_modulus_sze = modulus.size();
  parms.set_coeff_modulus(modulus);
  seal::SEALContext ecd_ms_context(parms, false, seal::sec_level_type::none);

  if (kUseModDownOptimization) {
    modulus.pop_back();
    if (field_bitlen > 64) {
      // Keep 3 primes are enough for FM128.
      modulus.pop_back();
    }
  }

  size_t dcd_modulus_sze = modulus.size();
  parms.set_use_special_prime(false);
  parms.set_coeff_modulus(modulus);
  seal::SEALContext dcd_ms_context(parms, false, seal::sec_level_type::none);

  seal_cntxts_.emplace(field_bitlen, this_context);
  secret_keys_.emplace(field_bitlen, rlwe_sk);
  peer_pub_keys_.emplace(field_bitlen, peer_public_key);

  ecd_mswh_.emplace(field_bitlen,
                    new ModulusSwitchHelper(ecd_ms_context, field_bitlen));
  dcd_mswh_.emplace(field_bitlen,
                    new ModulusSwitchHelper(dcd_ms_context, field_bitlen));

  decryptors_.emplace(field_bitlen,
                      new seal::Decryptor(*this_context, *rlwe_sk));
  if (need_galois_keys) {
    LazyInitGaloisKey(field_bitlen);
  }

  if (lctx_->Rank() == 0) {
    SPDLOG_INFO(
        "CheetahDot uses {}@{} modulus {} degree for {} bit ring (packing={})",
        enc_modulus_sze, dcd_modulus_sze, parms.poly_modulus_degree(),
        field_bitlen, need_galois_keys ? "enabled" : "disabled");
  }
}

void CheetahDot::Impl::doDotOLEReceiverRecvStep(const NdArrayRef &prv_mat,
                                                const Shape3D &dim3,
                                                bool is_self_lhs,
                                                CipherPackingType cptype,
                                                absl::Span<RLWECt> result_cts,
                                                yacl::link::Context *conn,
                                                size_t *num_packed) {
  int next_rank = conn->NextRank();
  auto eltype = prv_mat.eltype();
  auto field = eltype.template as<Ring2k>()->field();
  const size_t field_bitlen = SizeOf(field) * 8;

  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  const auto &this_ecd_msh = *ecd_mswh_.find(field_bitlen)->second;
  bool disable_pack = cptype == CipherPackingType::none;

  MatMatProtocol matmat_prot(this_context, this_ecd_msh, disable_pack);

  MatMatProtocol::Meta meta;
  meta.dims = dim3;
  auto subshape = matmat_prot.GetSubMatShape(meta);
  const size_t lhs_n = matmat_prot.GetLeftSize(meta, subshape);
  const size_t rhs_n = matmat_prot.GetRightSize(meta, subshape);
  const size_t out_n = matmat_prot.GetOutSize(meta, subshape);
  SPU_ENFORCE_EQ(out_n, result_cts.size());

  // 1. launch IO task to recv ct from peer
  std::vector<RLWECt> enc_mat(is_self_lhs ? rhs_n : lhs_n);
  auto io_task = std::async(std::launch::async, [&]() {
    for (auto &ct : enc_mat) {
      auto ct_s = conn->Recv(next_rank, "recv encrypted mat");
      DecodeSEALObject(ct_s, this_context, &ct);
    }
  });

  // 2+3. ENCODE AND MULTIPLY ONE SLICE AT A TIME (campaign step
  // b1_dot_encode_chunk).
  //
  // The stock code sizes `plain_mat` at the WHOLE encoded side and fills it
  // before the first multiplication reads any of it. That array is the largest
  // object in this system's measured baseline: for BERT-base's FFN
  // up-projection the encoded side is 12 x 3072 = 36,864 plaintexts of
  // 8192 x 3 x 8 B = 7.25 GB, and because SEAL's memory pool retains what it
  // hands out, the whole run's resident peak ratchets up to it.
  //
  // Nothing needs them to coexist. Each output block accumulates over the INNER
  // block index only, so an outer slice of the encoded side can be built, used
  // and dropped before the next is built. This is a change to WHEN a plaintext
  // exists, not to what is computed or to what is sent: same terms, same order
  // per accumulator, same transcript, bit-identical result.
  //
  // THE SLICE IS SIZED IN BYTES, NOT IN BLOCKS. The encoded side's shape varies
  // by more than two orders of magnitude across the operators of one layer
  // (12 polynomials for a QK^T, 36,864 for an FFN up-projection), so a fixed
  // block count would be most of the peak on one and pointless overhead on
  // another. The budget below is the resident plaintext bytes a slice may hold;
  // the block count follows from the polynomial size of the ring in use.
  //
  // 256 MiB: comfortably above the largest single INNER stripe this workload
  // produces, so no operator is forced below one slice and the parallel_for
  // inside a slice keeps thousands of jobs for 16 threads, and far below the
  // 7.25 GB it replaces.
  // 64 MiB (campaign step b4_slice_budget_64m; b1 to b3 ran at 256 MiB). First step of this
  // line's PAID phase: rule 6 ranks paid fixes by memory saved per unit of runtime spent, and
  // this is the best ratio available at 178 MiB per percent of wall clock. Measured beforehand as
  // a diagnostic on the b2 image (off-path-runs/x2_slice_budget_64m): -236,444 kB for +1.3%.
  const int64_t kSliceBudgetBytes = 64L << 20;

  const bool slice_lhs = is_self_lhs;
  const int64_t num_outer = matmat_prot.GetOuterBlocks(meta, slice_lhs);
  const int64_t num_inner = matmat_prot.GetInnerBlocks(meta);
  SPU_ENFORCE_EQ(static_cast<int64_t>(is_self_lhs ? lhs_n : rhs_n),
                 num_outer * num_inner);

  // One encoded polynomial is poly_degree x (RNS primes at the ENCODE level) x
  // 8 B. Taken from the context rather than assumed, because the two rings this
  // campaign can run (FM64 and FM32) differ in both factors.
  const int64_t poly_bytes = [&]() {
    auto cdata = this_context.first_context_data();
    return static_cast<int64_t>(cdata->parms().poly_modulus_degree()) *
           static_cast<int64_t>(cdata->parms().coeff_modulus().size()) *
           static_cast<int64_t>(sizeof(uint64_t));
  }();

  int64_t outer_per_slice = kSliceBudgetBytes / (num_inner * poly_bytes);
  outer_per_slice = std::max<int64_t>(1, std::min(outer_per_slice, num_outer));

  // The output accumulators are released ONCE, here, and not per slice: the
  // slices accumulate into them.
  for (auto &ct : result_cts) {
    ct.release();
  }

  // The receive of the peer's ciphertexts overlapped the stock encode of the
  // whole side. It cannot overlap a slice loop that consumes those very
  // ciphertexts, so it is awaited first. What is lost is the overlap with the
  // SMALLER side's transfer (12 ciphertexts for the FFN case), not with the
  // encode work itself, which is unchanged in total.
  io_task.get();

  // PACK EACH SLICE'S RESULTS AS SOON AS THEY ARE FINAL, AND FREE THE RAW ONES (campaign step
  // b2_result_ct_pack_free).
  //
  // What b1 left behind. `PackingWithModulusDrop` folds each group of `subshape[1]` = 64 result
  // ciphertexts into ONE, writing the packed ciphertext to index i/64 of the same array. The other
  // 63 are dead from that moment and nothing releases them: for the FFN up-projection that is 3024
  // of 3072 ciphertexts at 393,216 B each, 1.19 GB of ciphertexts the protocol has finished with.
  // They were the largest remaining occupant of SEAL's pool at b1's peak.
  //
  // Releasing them where the stock code packs would not move the peak, because by then all 3072
  // already exist -- the pool has already grown. What lowers the peak is packing EARLIER, and the
  // slice loop is what makes that possible: it walks k in ascending order and covers every j
  // within a slice, so out[i,k] for k below `oend` is final and will never be touched again.
  //
  // In-place packing stays safe under this reordering because the destination index i/64 always
  // lies BELOW i: packed results occupy [0, k/64) while the raw ciphertexts of the current slice
  // occupy [obgn, oend), and k/64 < k, so the two regions cannot overlap.
  //
  // GUARDED, because the flat output index is i*nb2 + k and a slice over k is only a contiguous
  // run of that index when there is a single row block. With more than one, packing groups would
  // straddle rows that are not final yet. The batched caller is excluded for the same kind of
  // reason: it packs across a span covering every batch, whose group boundaries need not align
  // with one batch's results. Both fall back to packing in the send step, unchanged.
  const int64_t num_row_blocks = matmat_prot.GetOuterBlocks(meta, /*for_lhs*/ true);
  const size_t gap = static_cast<size_t>(subshape[1]);
  const bool fuse_pack = (num_packed != nullptr) &&
                         (cptype == CipherPackingType::rlwes) && !slice_lhs &&
                         (num_row_blocks == 1) && (gap > 0);
  size_t packed_upto = 0;  // every result below this index has been folded away

  std::unique_ptr<PackingHelper> pack_helper;
  if (fuse_pack) {
    const auto &this_galois_key =
        *(peer_galois_keys_.find(field_bitlen)->second);
    pack_helper = std::make_unique<PackingHelper>(
        gap, this_ecd_msh.coeff_modulus_size(), this_galois_key, this_context);
  }

  // Fold every group whose 64 raw ciphertexts are final, then release those raw ciphertexts. The
  // group's own destination is skipped: it now holds the packed result.
  auto drain_packed = [&](size_t final_upto, bool flush_partial) {
    if (!fuse_pack) return;
    while (packed_upto < final_upto) {
      const size_t remaining = final_upto - packed_upto;
      if (remaining < gap && !flush_partial) break;
      const size_t this_batch = std::min(remaining, gap);
      const size_t packed_idx = packed_upto / gap;
      pack_helper->PackingWithModulusDrop(
          result_cts.subspan(packed_upto, this_batch),
          result_cts[packed_idx]);
      for (size_t j = packed_upto; j < packed_upto + this_batch; ++j) {
        if (j != packed_idx) {
          result_cts[j].release();
        }
      }
      packed_upto += this_batch;
    }
  };

  std::vector<RLWEPt> plain_slice;
  for (int64_t obgn = 0; obgn < num_outer; obgn += outer_per_slice) {
    const int64_t oend = std::min(num_outer, obgn + outer_per_slice);
    plain_slice.resize((oend - obgn) * num_inner);

    if (slice_lhs) {
      matmat_prot.EncodeLHSSlice(prv_mat, meta, false, obgn, oend,
                                 absl::MakeSpan(plain_slice));
    } else {
      matmat_prot.EncodeRHSSlice(prv_mat, meta, false, obgn, oend,
                                 absl::MakeSpan(plain_slice));
    }

    yacl::parallel_for(0, plain_slice.size(), [&](size_t bgn, size_t end) {
      for (size_t i = bgn; i < end; ++i) {
        NttInplace(plain_slice[i], this_context);
      }
    });

    if (slice_lhs) {
      matmat_prot.ComputeLHSSlice(plain_slice, enc_mat, meta, obgn, oend,
                                  result_cts);
    } else {
      matmat_prot.ComputeRHSSlice(enc_mat, plain_slice, meta, obgn, oend,
                                  result_cts);
    }

    // Hand the slice's plaintexts back before building the next one. `resize`
    // alone would keep the capacity, and a seal::Plaintext keeps its pool
    // allocation until it is destroyed, so the whole point of slicing would be
    // lost to the vector's own growth policy.
    std::vector<RLWEPt>().swap(plain_slice);

    // Everything below `oend` is final now, so fold away every complete group.
    drain_packed(static_cast<size_t>(oend), /*flush_partial*/ false);
  }

  // The last group may be short; it can only be packed once the loop is over.
  drain_packed(result_cts.size(), /*flush_partial*/ true);

  if (num_packed != nullptr) {
    *num_packed = fuse_pack ? CeilDiv(result_cts.size(), gap) : 0;
  }
}

NdArrayRef CheetahDot::Impl::doDotOLEReceiverSendStep(
    FieldType field, size_t batch_size, MatMatProtocol::Meta meta,
    absl::Span<RLWECt> ct_array_to_pack, CipherPackingType cptype,
    yacl::link::Context *conn, size_t bytes_recv, size_t pre_packed) {
  int next_rank = conn->NextRank();
  const size_t field_bitlen = SizeOf(field) * 8;
  const auto &this_public_key = *(peer_pub_keys_.find(field_bitlen)->second);
  const auto &this_dcd_msh = *dcd_mswh_.find(field_bitlen)->second;
  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  const auto &this_ecd_msh = *ecd_mswh_.find(field_bitlen)->second;
  bool disable_pack = cptype == CipherPackingType::none;

  MatMatProtocol matmat_prot(this_context, this_ecd_msh, disable_pack);

  auto subshape = matmat_prot.GetSubMatShape(meta);

  size_t out_n = ct_array_to_pack.size();
  size_t num_ct_response = 0;

  yacl::ElapsedTimer pack_timer;
  if (cptype == CipherPackingType::none) {
    SPU_ENFORCE(batch_size == 1, "impossible dispatch here");
    num_ct_response = out_n;
  } else if (cptype == CipherPackingType::rlwes) {
    // BumbleBee's interleave
    const size_t gap = subshape[1];
    const size_t pack_stride = gap;

    if (pre_packed > 0) {
      // The recv step already folded these, group by group, as each became final, and released
      // the raw ciphertexts it consumed (campaign step b2_result_ct_pack_free). Repeating the
      // loop here would pack packed ciphertexts. The count is re-derived rather than trusted, so
      // a disagreement is a build-time-visible bug and not a silently short response.
      SPU_ENFORCE_EQ(pre_packed, CeilDiv(out_n, pack_stride));
      num_ct_response = pre_packed;
    } else {
      const auto &this_galois_key =
          *(peer_galois_keys_.find(field_bitlen)->second);

      PackingHelper pack_helper(gap, this_ecd_msh.coeff_modulus_size(),
                                this_galois_key, this_context);

      for (size_t i = 0; i < out_n; i += pack_stride) {
        size_t this_batch = std::min(out_n - i, pack_stride);
        size_t packed_idx = i / pack_stride;
        pack_helper.PackingWithModulusDrop(
            ct_array_to_pack.subspan(i, this_batch),
            ct_array_to_pack[packed_idx]);
      }

      num_ct_response = CeilDiv(out_n, pack_stride);
    }
  } else {
    // Chen Hao et al's PackLWEs
    SPU_ENFORCE(batch_size == 1, "not implemented yet");
    const auto &this_galois_key =
        *(peer_galois_keys_.find(field_bitlen)->second);

    // Drop some modulus before packing
    yacl::parallel_for(
        0, ct_array_to_pack.size(), [&](int64_t bgn, int64_t end) {
          for (int64_t i = bgn; i < end; ++i) {
            InvNttInplace(ct_array_to_pack[i], this_context);
            ModulusSwtichInplace(ct_array_to_pack[i],
                                 this_dcd_msh.coeff_modulus_size(),
                                 this_context);
          }
        });

    // Extract Phantom LWECt from RLWEs
    size_t out_numel = meta.dims[0] * meta.dims[2];
    size_t aligned_onumel = absl::bit_ceil(out_numel);
    std::vector<PhantomLWECt> _lwes(aligned_onumel);
    auto lwes = absl::MakeSpan(_lwes);
    matmat_prot.WrapPhantomLWEs(meta, ct_array_to_pack,
                                lwes.subspan(0, out_numel));

    size_t pack_stride = lwes[0].poly_modulus_degree();
    std::vector<RLWECt> packed_lwes(CeilDiv(lwes.size(), pack_stride));

    // NOTE: we can not modify the ct_array_to_pack inplace because it wraps the
    // PhantomLWECt
    PackLWEs(lwes, this_galois_key, this_context, absl::MakeSpan(packed_lwes));

    num_ct_response = packed_lwes.size();

    // copy back
    for (size_t i = 0; i < num_ct_response; ++i) {
      ct_array_to_pack[i] = packed_lwes[i];
    }
  }
  double pack_time = pack_timer.CountMs();

  // 4. Random masking to conver HE to AShr
  std::vector<RLWEPt> rnd_polys(num_ct_response);
  H2A({ct_array_to_pack.data(), num_ct_response}, absl::MakeSpan(rnd_polys),
      this_dcd_msh.coeff_modulus_size(), this_public_key, this_context);

  if (cptype == CipherPackingType::none) {
    // If no packing, then clean up un-used coefficients
    // NOTE(lwj): we place Extract **after** H2A for a smaller communication
    matmat_prot.ExtractLWEsInplace(meta, ct_array_to_pack);
  }

  size_t bytes_sent = conn->GetStats()->sent_bytes;
  for (size_t i = 0; i < num_ct_response; ++i) {
    conn->SendAsync(next_rank, EncodeSEALObject(ct_array_to_pack[i]), "");
  }
  bytes_sent = conn->GetStats()->sent_bytes - bytes_sent;

  SPDLOG_INFO(
      "{}@{}x{}x{} => {}x{}x{} Recv {} MiB, Response {} MiB Pack {} ms ({})",
      batch_size, meta.dims[0], meta.dims[1], meta.dims[2], subshape[0],
      subshape[1], subshape[2],
      std::roundf(bytes_recv / 1024. / 1024. * 1000) / 1000.,
      std::roundf(bytes_sent / 1024. / 1024. * 1000) / 1000.,
      std::roundf(pack_time * 1000) / 1000., ToString(cptype));

  switch (cptype) {
    case CipherPackingType::none:
      return matmat_prot.ParseResult(
          field, meta, absl::MakeConstSpan(rnd_polys), this_dcd_msh);
    case CipherPackingType::lwes:
      return matmat_prot.ParsePackLWEsResult(
          field, meta, absl::MakeConstSpan(rnd_polys), this_dcd_msh);
    case CipherPackingType::rlwes:
    default:
      return matmat_prot.ParseBatchPackedResult(field, batch_size, meta,
                                                absl::MakeConstSpan(rnd_polys),
                                                this_dcd_msh);
  }
}

void CheetahDot::Impl::doDotOLESenderSendStep(const NdArrayRef &prv_mat,
                                              const Shape3D &dim3,
                                              bool is_self_lhs,
                                              CipherPackingType cptype,
                                              yacl::link::Context *conn) {
  int next_rank = conn->NextRank();
  auto eltype = prv_mat.eltype();
  auto field = eltype.template as<Ring2k>()->field();
  const size_t field_bitlen = SizeOf(field) * 8;

  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  auto &this_secret_key = *secret_keys_.find(field_bitlen)->second;
  auto &this_ecd_msh = *ecd_mswh_.find(field_bitlen)->second;
  bool disable_pack = cptype == CipherPackingType::none;
  MatMatProtocol matmat_prot(this_context, this_ecd_msh, disable_pack);

  MatMatProtocol::Meta meta;
  meta.dims = dim3;
  auto subshape = matmat_prot.GetSubMatShape(meta);

  const size_t lhs_n = matmat_prot.GetLeftSize(meta, subshape);
  const size_t rhs_n = matmat_prot.GetRightSize(meta, subshape);

  std::vector<RLWEPt> _encoded_mat(is_self_lhs ? lhs_n : rhs_n);
  auto encoded_mat = absl::MakeSpan(_encoded_mat);
  if (is_self_lhs) {
    matmat_prot.EncodeLHS(prv_mat, meta, true, encoded_mat);
  } else {
    matmat_prot.EncodeRHS(prv_mat, meta, true, encoded_mat);
  }

  size_t num_ct_to_send = encoded_mat.size();
  for (size_t i = 0; i < num_ct_to_send; i += kCtAsyncParallel) {
    size_t this_batch = std::min(num_ct_to_send - i, kCtAsyncParallel);
    std::vector<RLWECt> enc_mat(this_batch);
    std::vector<yacl::Buffer> ct_s(this_batch);

    SymmetricRLWEEncrypt(this_secret_key, this_context,
                         encoded_mat.subspan(i, this_batch),
                         /*ntt*/ true,
                         /*seed*/ true, absl::MakeSpan(enc_mat));

    yacl::parallel_for(0, this_batch, [&](size_t bgn, size_t end) {
      for (size_t j = bgn; j < end; ++j) {
        ct_s[j] = EncodeSEALObject(enc_mat[j]);
      }
    });

    for (size_t j = 1; j < this_batch; ++j) {
      conn->SendAsync(next_rank, ct_s[j - 1], "send encrypted mat");
    }
    conn->Send(next_rank, ct_s[this_batch - 1], "send encrypted mat");
  }
}

NdArrayRef CheetahDot::Impl::doDotOLESenderRecvStep(FieldType field,
                                                    size_t batch_size,
                                                    MatMatProtocol::Meta meta,
                                                    size_t num_ct_to_recv,
                                                    CipherPackingType cptype,
                                                    yacl::link::Context *conn) {
  SPU_ENFORCE(batch_size > 0);
  int next_rank = conn->NextRank();
  const size_t field_bitlen = SizeOf(field) * 8;

  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  auto &this_decryptor = decryptors_.find(field_bitlen)->second;
  auto &this_ecd_msh = *ecd_mswh_.find(field_bitlen)->second;
  auto &this_dcd_msh = *dcd_mswh_.find(field_bitlen)->second;

  MatMatProtocol matmat_prot(this_context, this_ecd_msh,
                             cptype == CipherPackingType::none);

  std::vector<RLWECt> recv_ct(kCtAsyncParallel);
  std::vector<RLWEPt> result_poly(num_ct_to_recv);

  for (size_t i = 0; i < num_ct_to_recv; i += kCtAsyncParallel) {
    size_t this_batch = std::min(num_ct_to_recv - i, kCtAsyncParallel);
    for (size_t j = 0; j < this_batch; ++j) {
      auto ct_s = conn->Recv(next_rank, "recv result mat");
      DecodeSEALObject(ct_s, this_context, &recv_ct[j]);
    }

    yacl::parallel_for(0, this_batch, [&](size_t bgn, size_t end) {
      for (size_t j = bgn; j < end; ++j) {
        NttInplace(recv_ct[j], this_context);
        this_decryptor->decrypt(recv_ct[j], result_poly[i + j]);
        // non-ntt form for ParseResult
        InvNttInplace(result_poly[i + j], this_context);
      }
    });
  }

  switch (cptype) {
    case CipherPackingType::none:
      return matmat_prot.ParseResult(
          field, meta, absl::MakeConstSpan(result_poly), this_dcd_msh);
    case CipherPackingType::lwes:
      return matmat_prot.ParsePackLWEsResult(
          field, meta, absl::MakeConstSpan(result_poly), this_dcd_msh);
    case CipherPackingType::rlwes:
    default:
      return matmat_prot.ParseBatchPackedResult(
          field, batch_size, meta, absl::MakeConstSpan(result_poly),
          this_dcd_msh);
  }
}

NdArrayRef CheetahDot::Impl::DotOLE(const NdArrayRef &prv_mat,
                                    yacl::link::Context *conn,
                                    const Shape3D &dim3, bool is_self_lhs) {
  if (conn == nullptr) {
    conn = lctx_.get();
  }
  auto eltype = prv_mat.eltype();
  SPU_ENFORCE(eltype.isa<Ring2k>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(prv_mat.numel() > 0 && prv_mat.ndim() == 2);

  if (is_self_lhs) {
    SPU_ENFORCE_EQ(prv_mat.numel(), dim3[0] * dim3[1]);
  } else {
    SPU_ENFORCE_EQ(prv_mat.numel(), dim3[1] * dim3[2]);
  }

  return doDotOLE(prv_mat, conn, dim3, is_self_lhs);
}

NdArrayRef CheetahDot::Impl::BatchDotOLE(const NdArrayRef &prv_mat,
                                         yacl::link::Context *conn,
                                         const Shape4D &dim4,
                                         bool is_self_lhs) {
  if (conn == nullptr) {
    conn = lctx_.get();
  }
  auto eltype = prv_mat.eltype();
  SPU_ENFORCE(eltype.isa<Ring2k>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(prv_mat.numel() > 0 && prv_mat.ndim() == 3);

  if (is_self_lhs) {
    SPU_ENFORCE_EQ(prv_mat.numel(), dim4[0] * dim4[1] * dim4[2]);
  } else {
    SPU_ENFORCE_EQ(prv_mat.numel(), dim4[0] * dim4[2] * dim4[3]);
  }

  auto field = eltype.template as<Ring2k>()->field();
  if (not IsPackingEnabled(8 * SizeOf(field))) {
    // Packing is not supported.
    // Thus just call multiple DotOLEs
    Shape3D dim3 = {dim4[1], dim4[2], dim4[3]};
    int64_t out_numel = dim4[1] * dim4[3];
    NdArrayRef out(eltype, {dim4[0] * out_numel});
    const auto &mat_shape = prv_mat.shape();

    for (int64_t b = 0; b < dim4[0]; ++b) {
      auto one_mat =
          prv_mat
              .slice({b, 0, 0}, {b + 1, mat_shape[1], mat_shape[2]}, {1, 1, 1})
              .reshape({mat_shape[1], mat_shape[2]});
      auto one_out = DotOLE(one_mat, conn, dim3, is_self_lhs);
      auto out_slice =
          out.slice({b * out_numel}, {b * out_numel + out_numel}, {1});
      pforeach(0, one_out.numel(), [&](int64_t i) {
        std::memcpy(&out_slice.at(i), &one_out.at(i), one_out.elsize());
      });
    }

    return out.reshape({dim4[0], dim4[1], dim4[3]});
  }

  return doBatchDotOLE(prv_mat, conn, dim4, is_self_lhs);
}

NdArrayRef CheetahDot::Impl::doBatchDotOLE(const NdArrayRef &prv_mat,
                                           yacl::link::Context *conn,
                                           const Shape4D &dim4,
                                           bool is_self_lhs) {
  auto eltype = prv_mat.eltype();
  auto field = eltype.template as<Ring2k>()->field();

  const size_t field_bitlen = SizeOf(field) * 8;
  size_t poly_deg = DecideSEALParameters(field_bitlen).poly_modulus_degree();

  const Shape3D dim3 = {dim4[1], dim4[2], dim4[3]};
  const size_t batch_size = dim4[0];
  MatMatProtocol::Meta meta = {.dims = dim3};

  CipherPackingType cptype = CipherPackingType::rlwes;
  auto subshape =
      MatMatProtocol::GetSubMatShape(meta, poly_deg, /*disable_pack*/ false);

  size_t blk0 = CeilDiv(meta.dims[0], subshape[0]);
  size_t blk1 = CeilDiv(meta.dims[1], subshape[1]);
  size_t blk2 = CeilDiv(meta.dims[2], subshape[2]);
  size_t lhs_poly_n = blk0 * blk1;
  size_t rhs_poly_n = blk1 * blk2;
  bool to_encrypt_lhs = lhs_poly_n <= rhs_poly_n;
  bool act_as_encryptor = (is_self_lhs ^ to_encrypt_lhs) == 0;

  LazyInit(field_bitlen, cptype != CipherPackingType::none);
  auto mat_shape = prv_mat.shape();

  if (act_as_encryptor) {
    for (int64_t b = 0; b < (int64_t)batch_size; ++b) {
      auto one_mat =
          prv_mat
              .slice({b, 0, 0}, {b + 1, mat_shape[1], mat_shape[2]}, {1, 1, 1})
              .reshape({mat_shape[1], mat_shape[2]});
      doDotOLESenderSendStep(one_mat, dim3, is_self_lhs, cptype, conn);
    }

    size_t num_ct_to_recv = 0;
    switch (cptype) {
      case CipherPackingType::rlwes:
        num_ct_to_recv = CeilDiv<size_t>(batch_size * blk0 * blk2, subshape[1]);
        break;
      default:
        num_ct_to_recv = blk0 * blk2;
    }
    return doDotOLESenderRecvStep(field, batch_size, meta, num_ct_to_recv,
                                  cptype, conn);
  }

  size_t num_ct_per_batch = blk0 * blk2;
  std::vector<RLWECt> _ct_array_to_pack(batch_size * num_ct_per_batch);
  auto ct_array_to_pack = absl::MakeSpan(_ct_array_to_pack);

  size_t bytes_recv = conn->GetStats()->recv_bytes;
  for (int64_t b = 0; b < (int64_t)batch_size; ++b) {
    auto one_mat =
        prv_mat.slice({b, 0, 0}, {b + 1, mat_shape[1], mat_shape[2]}, {1, 1, 1})
            .reshape({mat_shape[1], mat_shape[2]});
    doDotOLEReceiverRecvStep(
        one_mat, dim3, is_self_lhs, cptype,
        ct_array_to_pack.subspan(b * num_ct_per_batch, num_ct_per_batch), conn);
  }
  bytes_recv = conn->GetStats()->recv_bytes - bytes_recv;

  return doDotOLEReceiverSendStep(field, batch_size, meta, ct_array_to_pack,
                                  cptype, conn, bytes_recv);
}

NdArrayRef CheetahDot::Impl::doDotOLE(const NdArrayRef &prv_mat,
                                      yacl::link::Context *conn,
                                      const Shape3D &dim3, bool is_self_lhs) {
  auto eltype = prv_mat.eltype();
  auto field = eltype.template as<Ring2k>()->field();
  const size_t field_bitlen = SizeOf(field) * 8;
  size_t poly_deg = DecideSEALParameters(field_bitlen).poly_modulus_degree();

  MatMatProtocol::Meta meta = {.dims = dim3};

  // No cipher packing for small HE
  CipherPackingType cptype =
      (IsPackingEnabled(8 * SizeOf(field)) and not disable_pack_)
          ? CipherPackingType::rlwes
          : CipherPackingType::none;

  Shape3D subshape;
  size_t blk[3];
  if (cptype != CipherPackingType::none) {
    // attempt to calculate the cost with packing
    subshape = MatMatProtocol::GetSubMatShape(meta, poly_deg, false);
    for (int i : {0, 1, 2}) {
      blk[i] = CeilDiv(meta.dims[i], subshape[i]);
    }

    if (blk[0] * blk[2] <= 1) {
      // If there is only 1 resultant RLWE;
      // then we just skip any packing
      cptype = CipherPackingType::none;
    } else {
      // dynamic packing type
      double pack_rlwes_cost = subshape[1];
      double pack_lwes_cost = meta.dims[0] * meta.dims[2];
      cptype = pack_rlwes_cost < pack_lwes_cost ? CipherPackingType::rlwes
                                                : CipherPackingType::lwes;
    }
  }

  LazyInit(field_bitlen, cptype != CipherPackingType::none);

  // Update subshape
  subshape = MatMatProtocol::GetSubMatShape(meta, poly_deg,
                                            cptype == CipherPackingType::none);

  for (int i : {0, 1, 2}) {
    blk[i] = CeilDiv(meta.dims[i], subshape[i]);
  }

  size_t lhs_poly_n = blk[0] * blk[1];
  size_t rhs_poly_n = blk[1] * blk[2];
  bool to_encrypt_lhs = lhs_poly_n <= rhs_poly_n;
  bool act_as_encryptor = (is_self_lhs ^ to_encrypt_lhs) == 0;

  if (act_as_encryptor) {
    doDotOLESenderSendStep(prv_mat, dim3, is_self_lhs, cptype, conn);

    size_t num_ct_to_recv = 0;
    switch (cptype) {
      case CipherPackingType::rlwes:
        num_ct_to_recv = CeilDiv<size_t>(blk[0] * blk[2], subshape[1]);
        break;
      case CipherPackingType::lwes:
        num_ct_to_recv = CeilDiv<size_t>(meta.dims[0] * meta.dims[2], poly_deg);
        break;
      default:
      case CipherPackingType::none:
        num_ct_to_recv = blk[0] * blk[2];
        break;
    }

    return doDotOLESenderRecvStep(field, /*batch*/ 1, meta, num_ct_to_recv,
                                  cptype, conn)
        .reshape({dim3[0], dim3[2]});
  }

  std::vector<RLWECt> _ct_array_to_pack(blk[0] * blk[2]);
  auto ct_array_to_pack = absl::MakeSpan(_ct_array_to_pack);
  size_t bytes_recv = conn->GetStats()->recv_bytes;
  // Passing the pointer is what permits the recv step to fold each group of results as soon as it
  // is final and release the raw ciphertexts. Only this single-matmul path does so; the batched
  // one below keeps the stock behaviour, because its packing groups run across batches.
  size_t num_packed = 0;
  doDotOLEReceiverRecvStep(prv_mat, dim3, is_self_lhs, cptype, ct_array_to_pack,
                           conn, &num_packed);
  bytes_recv = conn->GetStats()->recv_bytes - bytes_recv;

  return doDotOLEReceiverSendStep(field, /*batch*/ 1, meta, ct_array_to_pack,
                                  cptype, conn, bytes_recv, num_packed)
      .reshape({dim3[0], dim3[2]});
}

//////////////////////////////////////////////

CheetahDot::CheetahDot(const std::shared_ptr<yacl::link::Context> &lctx,
                       bool disable_matmul_pack) {
  impl_ = std::make_unique<Impl>(lctx, disable_matmul_pack);
}

CheetahDot::~CheetahDot() = default;

void CheetahDot::LazyInitKeys(FieldType field) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->LazyInit(SizeOf(field) * 8,
                         impl_->IsPackingEnabled(SizeOf(field) * 8));
}

NdArrayRef CheetahDot::DotOLE(const NdArrayRef &inp, yacl::link::Context *conn,
                              const Shape3D &dim3, bool is_self_lhs) {
  SPU_ENFORCE(impl_ != nullptr);
  SPU_ENFORCE(conn != nullptr);
  return impl_->DotOLE(inp, conn, dim3, is_self_lhs);
}

NdArrayRef CheetahDot::DotOLE(const NdArrayRef &inp, const Shape3D &dim3,
                              bool is_self_lhs) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->DotOLE(inp, nullptr, dim3, is_self_lhs);
}

NdArrayRef CheetahDot::BatchDotOLE(const NdArrayRef &inp,
                                   yacl::link::Context *conn,
                                   const Shape4D &dim4, bool is_self_lhs) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->BatchDotOLE(inp, conn, dim4, is_self_lhs);
}

}  // namespace spu::mpc::cheetah
