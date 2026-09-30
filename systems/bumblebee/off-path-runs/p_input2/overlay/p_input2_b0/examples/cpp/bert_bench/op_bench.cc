// Instrumented BERT-base benchmark for BumbleBee (Cheetah 2PC) per-operator
// memory profiling. Uses SPU's C++ API directly -- no Python, JAX, or PPHLO.
//
// Each party runs as a SEPARATE PROCESS connected via brpc, so /proc/self/status
// captures per-party memory (client vs server).
//
// Usage:   ./op_bench --rank <0|1>
//
// Environment variables:
//   BB_N_LAYERS        -- number of transformer layers (default 12)
//   BB_SEQ_LEN         -- input sequence length (default 128)
//   BB_MAX_CONCURRENCY -- max parallel workers (default: hardware threads)
//
// Output markers (to stderr, same format as SHARK):
//   MEM|<party>|<layer>|<phase>|<op>|<rss_kb>|<timestamp_ms>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

#include "libspu/core/context.h"
#include "libspu/core/pt_buffer_view.h"
#include "libspu/core/value.h"

#include "libspu/mpc/cheetah/protocol.h"

#include "yacl/link/link.h"

#include "libspu/kernel/hal/constants.h"
#include "libspu/kernel/hal/polymorphic.h"
#include "libspu/kernel/hal/public_helper.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/kernel/hal/type_cast.h"

// BumbleBee-specific intrinsics (GeLU, neg_exp)
#include "libspu/kernel/hal/intrinsic/nn/activation.h"

#include "libspu/spu.pb.h"

namespace hal = spu::kernel::hal;
using spu::Value;

// --------------- Memory instrumentation (same as SHARK) ---------------

static long read_vmrss_kb() {
    FILE* fp = fopen("/proc/self/status", "r");
    if (!fp) return -1;
    char line[256];
    long rss = -1;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%ld", &rss);
            break;
        }
    }
    fclose(fp);
    return rss;
}

static long long now_ms() {
    auto tp = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(tp).count();
}

static void mem(int rank, const char* phase, const char* op, int layer_idx) {
    long rss = read_vmrss_kb();
    long long ts = now_ms();
    fprintf(stderr, "MEM|%d|%d|%s|%s|%ld|%lld\n",
            rank, layer_idx, phase, op, rss, ts);
    fflush(stderr);
}

// --------------- Gate observable ---------------
//
// Writes a revealed tensor where the harness asked for it (BB_GATE_FILE), one value per line, and
// prints a one-line checksum for every party.
//
// IT GOES TO ITS OWN FILE, NOT TO stderr, and that is not tidiness. brpc runs logging threads that
// write to fd 2 throughout the run, and stderr is unbuffered, so 98304 fprintf calls are 98304
// separate writes that another thread can splice a newline into. Measured on the first gate-check
// run: the observable arrived split, and recovering it with `grep ... | head -1` returned 19834 of
// 98304 values while every size and exit code looked entirely normal.
static void emit_gate(spu::SPUContext* ctx, const spu::Value& secret, int rank,
                      const char* tag, bool is_gate) {
    auto pub = secret.isPublic() ? secret : hal::reveal(ctx, secret);
    auto host = hal::dump_public_as<float>(ctx, pub);
    double checksum = 0.0;
    for (auto v : host) checksum += static_cast<double>(v);

    if (is_gate && rank == 0) {
        const char* gpath = getenv("BB_GATE_FILE");
        FILE* gf = (gpath != nullptr) ? fopen(gpath, "w") : nullptr;
        if (gf == nullptr) {
            fprintf(stderr, "GATEFAIL|rank=0|no writable BB_GATE_FILE\n");
        } else {
            for (auto v : host) fprintf(gf, "%.9e\n", static_cast<double>(v));
            fflush(gf);
            fclose(gf);
            fprintf(stderr, "GATE|%s|rank=0|n=%zu|file=%s\n", tag, host.size(), gpath);
        }
    }
    fprintf(stderr, "GATESUM|%s|rank=%d|n=%zu|%.9e\n", tag, rank, host.size(), checksum);
    fflush(stderr);
}

// --------------- Random tensor creation ---------------

static std::vector<float> rand_floats(size_t n, unsigned seed = 42) {
    std::vector<float> v(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& x : v) x = dist(rng);
    return v;
}

static Value make_secret(spu::SPUContext* ctx, int64_t rows, int64_t cols,
                         unsigned seed = 42) {
    auto data = rand_floats(rows * cols, seed);
    auto pub = hal::constant(ctx, data, spu::DT_F32, {rows, cols});
    return hal::seal(ctx, pub);
}

// --------------- Operators ---------------

// faithful_nonlinear: reduce (rows, cols) to (rows, 1) along the columns by halving; the sum uses
// local share additions, the maximum BumbleBee's comparison protocol through hal::max.
static Value row_reduce(spu::SPUContext* ctx, Value x, bool take_max) {
    const int64_t rows = x.shape()[0];
    while (x.shape()[1] > 1) {
        const int64_t c = x.shape()[1];
        const int64_t h = c / 2;
        auto a = hal::slice(ctx, x, {0, 0}, {rows, h}, {});
        auto b = hal::slice(ctx, x, {0, h}, {rows, 2 * h}, {});
        auto r = take_max ? hal::max(ctx, a, b) : hal::add(ctx, a, b);
        if (c % 2 != 0) {
            auto last = hal::slice(ctx, x, {0, c - 1}, {rows, c}, {});
            auto first = hal::slice(ctx, r, {0, 0}, {rows, 1}, {});
            auto f = take_max ? hal::max(ctx, first, last) : hal::add(ctx, first, last);
            r = (h > 1) ? hal::concatenate(ctx, {f, hal::slice(ctx, r, {0, 1}, {rows, h}, {})}, 1)
                        : f;
        }
        x = r;
    }
    return x;
}

// LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
// Simplified for profiling: does the key MPC operations (mul, add, rsqrt)
static Value layernorm(spu::SPUContext* ctx, const Value& x,
                       const Value& gamma, const Value& beta,
                       int rank, int layer_idx) {
    // faithful_nonlinear: (x - mean) / sqrt(var + eps) * gamma + beta over the hidden axis.
    const int64_t rows = x.shape()[0];
    const int64_t cols = x.shape()[1];
    auto inv_d = hal::constant(ctx, 1.0F / static_cast<float>(cols), spu::DT_F32, {rows, 1});
    auto mean = hal::mul(ctx, row_reduce(ctx, x, false), inv_d);
    auto xc = hal::sub(ctx, x, hal::broadcast_to(ctx, mean, x.shape()));
    auto var = hal::mul(ctx, row_reduce(ctx, hal::square(ctx, xc), false), inv_d);
    auto eps = hal::constant(ctx, 1e-5F, spu::DT_F32, {rows, 1});
    auto inv_std = hal::rsqrt(ctx, hal::add(ctx, var, eps));
    auto normed = hal::mul(ctx, xc, hal::broadcast_to(ctx, inv_std, x.shape()));
    auto g = hal::broadcast_to(ctx, gamma, x.shape());
    auto b = hal::broadcast_to(ctx, beta, x.shape());
    return hal::add(ctx, hal::mul(ctx, normed, g), b);
}

// Softmax: exp(x - max) / sum(exp(x - max))
// Uses BumbleBee's neg_exp intrinsic
static Value softmax_row(spu::SPUContext* ctx, const Value& x) {
    // faithful_nonlinear: exp(x - max) / sum(exp(x - max)) per row, as upstream's _softmax.
    auto mx = row_reduce(ctx, x, true);
    auto shifted = hal::sub(ctx, x, hal::broadcast_to(ctx, mx, x.shape()));
    auto nexp = spu::kernel::hal::intrinsic::nn::f_neg_exp_taylor(ctx, shifted);
    auto denom = hal::reciprocal(ctx, row_reduce(ctx, nexp, false));
    return hal::mul(ctx, nexp, hal::broadcast_to(ctx, denom, nexp.shape()));
}

// GeLU: uses BumbleBee's seg3 approximation intrinsic
static Value gelu(spu::SPUContext* ctx, const Value& x) {
    return spu::kernel::hal::intrinsic::nn::f_seg3_gelu(
        ctx, x, /*small_ring_compare=*/true);
}

// Linear: matmul(x, w) + bias
static Value linear(spu::SPUContext* ctx, const Value& x, const Value& w,
                    const Value& bias) {
    auto y = hal::matmul(ctx, x, w);
    auto b = hal::broadcast_to(ctx, bias, y.shape());
    auto z = hal::add(ctx, y, b);
    return z;
}

// --------------- Multi-Head Attention ---------------

static Value mha(spu::SPUContext* ctx, const Value& x, int layer_idx,
                 int rank, int n_heads, int head_dim, int seq_len,
                 const Value& w_qkv, const Value& b_qkv,
                 const Value& w_proj, const Value& b_proj) {
    mem(rank, "mha", "begin", layer_idx);

    // QKV projection: (seq, 768) x (768, 2304)
    auto qkv = linear(ctx, x, w_qkv, b_qkv);
    mem(rank, "mha", "after_qkv_linear", layer_idx);

    // Split Q, K, V using slice
    int hidden = n_heads * head_dim;
    auto q = hal::slice(ctx, qkv, {0, 0}, {seq_len, hidden}, {});
    auto k = hal::slice(ctx, qkv, {0, hidden}, {seq_len, 2 * hidden}, {});
    auto v = hal::slice(ctx, qkv, {0, 2 * hidden}, {seq_len, 3 * hidden}, {});

    // Per-head attention (profile head 0 in detail)
    std::vector<Value> head_outputs;
    for (int h = 0; h < n_heads; h++) {
        int start = h * head_dim;
        int end = (h + 1) * head_dim;

        auto qi = hal::slice(ctx, q, {0, start}, {seq_len, end}, {});
        auto ki = hal::slice(ctx, k, {0, start}, {seq_len, end}, {});
        auto vi = hal::slice(ctx, v, {0, start}, {seq_len, end}, {});

        // Q * K^T: (seq, 64) x (64, seq) -> (seq, seq)
        auto kt = hal::transpose(ctx, ki);
        auto qk = hal::matmul(ctx, qi, kt);
        // faithful_nonlinear: scale the scores by 1/sqrt(d_k) = 1/8 before the softmax.
        qk = hal::mul(ctx, qk, hal::constant(ctx, 0.125F, spu::DT_F32, qk.shape()));
        if (h == 0) mem(rank, "mha", "after_qk_matmul", layer_idx);

        // Softmax
        auto attn = softmax_row(ctx, qk);
        if (h == 0) mem(rank, "mha", "after_softmax", layer_idx);

        // Attn * V: (seq, seq) x (seq, 64) -> (seq, 64)
        auto av = hal::matmul(ctx, attn, vi);
        if (h == 0) mem(rank, "mha", "after_av_matmul", layer_idx);

        head_outputs.push_back(av);
    }

    // Concatenate heads: concat along last axis -> (seq, 768)
    auto concat = head_outputs[0];
    for (int h = 1; h < n_heads; h++) {
        concat = hal::concatenate(ctx, {concat, head_outputs[h]}, 1);
    }
    mem(rank, "mha", "after_all_heads", layer_idx);

    // Output projection: (seq, 768) x (768, 768)
    auto proj = linear(ctx, concat, w_proj, b_proj);
    mem(rank, "mha", "after_proj_linear", layer_idx);

    return proj;
}

// --------------- Feed-Forward Network ---------------

static Value ffn(spu::SPUContext* ctx, const Value& x, int layer_idx,
                 int rank, const Value& w_up, const Value& b_up,
                 const Value& w_down, const Value& b_down) {
    mem(rank, "ffn", "begin", layer_idx);

    // Up projection: (seq, 768) x (768, 3072)
    auto up = linear(ctx, x, w_up, b_up);
    mem(rank, "ffn", "after_up_linear", layer_idx);

    // GeLU activation (BumbleBee intrinsic)
    auto act = gelu(ctx, up);
    mem(rank, "ffn", "after_gelu", layer_idx);

    // Down projection: (seq, 3072) x (3072, 768)
    auto down = linear(ctx, act, w_down, b_down);
    mem(rank, "ffn", "after_down_linear", layer_idx);

    return down;
}

// --------------- Full transformer layer ---------------

static Value bert_layer(spu::SPUContext* ctx, const Value& x, int layer_idx,
                        int rank, int n_heads, int head_dim, int seq_len,
                        int hidden, int interm,
                        const Value& w_qkv, const Value& b_qkv,
                        const Value& w_proj, const Value& b_proj,
                        const Value& ln1_g, const Value& ln1_b,
                        const Value& w_up, const Value& b_up,
                        const Value& w_down, const Value& b_down,
                        const Value& ln2_g, const Value& ln2_b) {
    mem(rank, "layer", "begin", layer_idx);

    // Pre-LayerNorm
    auto normed = layernorm(ctx, x, ln1_g, ln1_b, rank, layer_idx);

    // Multi-Head Attention
    auto attn_out = mha(ctx, normed, layer_idx, rank, n_heads, head_dim,
                        seq_len, w_qkv, b_qkv, w_proj, b_proj);

    // Residual connection
    auto residual1 = hal::add(ctx, x, attn_out);
    mem(rank, "layer", "after_mha_residual", layer_idx);

    // Pre-LayerNorm
    auto normed2 = layernorm(ctx, residual1, ln2_g, ln2_b, rank, layer_idx);

    // FFN
    auto ffn_out = ffn(ctx, normed2, layer_idx, rank, w_up, b_up,
                       w_down, b_down);

    // Residual connection
    auto residual2 = hal::add(ctx, residual1, ffn_out);
    mem(rank, "layer", "after_ffn_residual", layer_idx);

    return residual2;
}

// --------------- Main ---------------

int main(int argc, char** argv) {
    int n_layers = 12;
    int seq_len = 128;
    int n_heads = 12;
    int hidden = 768;
    int interm = 3072;
    int head_dim = hidden / n_heads;  // 64

    int max_concurrency = 0;  // 0 = hardware default

    if (auto* e = getenv("BB_N_LAYERS")) n_layers = atoi(e);
    if (auto* e = getenv("BB_SEQ_LEN")) seq_len = atoi(e);
    if (auto* e = getenv("BB_MAX_CONCURRENCY")) max_concurrency = atoi(e);

    // Parse --rank argument
    int rank = -1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
            rank = atoi(argv[i + 1]);
        }
    }
    if (rank < 0 || rank > 1) {
        fprintf(stderr, "Usage: %s --rank <0|1>\n", argv[0]);
        return 1;
    }

    fprintf(stderr, "CONFIG|rank=%d|n_layers=%d|seq_len=%d|hidden=%d|heads=%d|max_concurrency=%d\n",
            rank, n_layers, seq_len, hidden, n_heads, max_concurrency);

    // SETUP MARKER, not a lever. Every plaintext this benchmark feeds the protocol comes from
    // std::mt19937 with a seed fixed in the source (rand_floats/make_secret), so the INPUT side is
    // reproducible by construction and the harness can prove that the binary it ran is the one
    // that says so. Whether the PROTOCOL is reproducible is a separate question, answered by
    // running the system twice unchanged and comparing GATE|logits| byte for byte; the secret
    // shares and the OT/HE randomness are drawn inside SPU and yacl, not here.
    //
    // Fixed inputs exist so that the gate can exist. They are not a deployment configuration.
    fprintf(stderr, "SEEDED|mt19937|weights=100..1200|input=7777|rank=%d\n", rank);
    fflush(stderr);

    // Create network link context (two separate processes via brpc)
    yacl::link::ContextDesc desc;
    desc.parties.push_back({"party0", "127.0.0.1:9530"});
    desc.parties.push_back({"party1", "127.0.0.1:9531"});
    auto lctx = yacl::link::FactoryBrpc().CreateContext(desc, rank);

    {
        spu::RuntimeConfig conf;
        conf.set_protocol(spu::ProtocolKind::CHEETAH);
        conf.set_field(spu::FieldType::FM64);
        conf.set_fxp_fraction_bits(16);
        conf.set_fxp_exp_iters(4);
        conf.set_fxp_div_goldschmidt_iters(2);
        conf.set_fxp_log_iters(3);
        conf.set_enable_pphlo_profile(true);
        conf.set_enable_hal_profile(true);
        if (max_concurrency > 0) {
            conf.set_max_concurrency(max_concurrency);
        }
        auto* cheetah_conf = conf.mutable_cheetah_2pc_config();
        cheetah_conf->set_ot_kind(spu::CheetahOtKind::YACL_Ferret);
        cheetah_conf->set_enable_mul_lsb_error(true);

        auto ctx = spu::mpc::makeCheetahProtocol(conf, lctx);
        int rank = lctx->Rank();

        mem(rank, "main", "after_init", -1);

        // Create random weights for all layers
        // (dimensions determine memory, not values)
        struct LayerWeights {
            Value w_qkv, b_qkv, w_proj, b_proj;
            Value ln1_g, ln1_b;
            Value w_up, b_up, w_down, b_down;
            Value ln2_g, ln2_b;
        };

        std::vector<LayerWeights> layers(n_layers);
        for (int i = 0; i < n_layers; i++) {
            layers[i].w_qkv = make_secret(ctx.get(), hidden, 3 * hidden, 100 + i);
            layers[i].b_qkv = make_secret(ctx.get(), 1, 3 * hidden, 200 + i);
            layers[i].w_proj = make_secret(ctx.get(), hidden, hidden, 300 + i);
            layers[i].b_proj = make_secret(ctx.get(), 1, hidden, 400 + i);
            layers[i].ln1_g = make_secret(ctx.get(), 1, hidden, 500 + i);
            layers[i].ln1_b = make_secret(ctx.get(), 1, hidden, 600 + i);
            layers[i].w_up = make_secret(ctx.get(), hidden, interm, 700 + i);
            layers[i].b_up = make_secret(ctx.get(), 1, interm, 800 + i);
            layers[i].w_down = make_secret(ctx.get(), interm, hidden, 900 + i);
            layers[i].b_down = make_secret(ctx.get(), 1, hidden, 1000 + i);
            layers[i].ln2_g = make_secret(ctx.get(), 1, hidden, 1100 + i);
            layers[i].ln2_b = make_secret(ctx.get(), 1, hidden, 1200 + i);
        }
        mem(rank, "main", "after_model_input", -1);

        // Create input embedding (random, as if looked up by client)
        Value x = make_secret(ctx.get(), seq_len, hidden, 7777);
        mem(rank, "main", "after_embedding", -1);

        // Run transformer layers
        for (int i = 0; i < n_layers; i++) {
            fprintf(stderr, "# Layer %d/%d (rank %d)\n", i + 1, n_layers, rank);
            fflush(stderr);

            x = bert_layer(ctx.get(), x, i, rank,
                           n_heads, head_dim, seq_len, hidden, interm,
                           layers[i].w_qkv, layers[i].b_qkv,
                           layers[i].w_proj, layers[i].b_proj,
                           layers[i].ln1_g, layers[i].ln1_b,
                           layers[i].w_up, layers[i].b_up,
                           layers[i].w_down, layers[i].b_down,
                           layers[i].ln2_g, layers[i].ln2_b);

            // THE GATE OBSERVABLE IS THE FIRST LAYER'S OUTPUT, and that is a measured decision
            // rather than a convenient one. See emit_gate() and the MANIFEST: the 12-layer output
            // of this benchmark is in fixed-point ring WRAPAROUND, so two unchanged runs of it
            // disagree by O(1) even though the protocol error is ~1 ULP. Measured on this image:
            //   N=1   two runs   1.668388728e+05 vs 1.668378700e+05   6.0e-6 relative
            //   N=12  two runs  -1.248762345e+12 vs -3.709103419e+11  chaotic
            // Layer 1 is the deepest point at which this benchmark's arithmetic is still
            // numerically meaningful, so it is where a reduction can be checked against the
            // baseline at all.
            if (i == 0) {
                emit_gate(ctx.get(), x, rank, "layer1", true);
            }
        }

        // Reveal output. Kept because it is what the stock benchmark does and therefore part of
        // the measured work, but its VALUE is evidence, not the gate: at 12 layers it is wrapped.
        auto result = hal::reveal(ctx.get(), x);
        mem(rank, "main", "after_output", -1);
        emit_gate(ctx.get(), result, rank, "final", false);

        // An exact, data-independent invariant beside the numeric one. The campaign's standing
        // check is two mechanisms for one quantity: a change to the protocol's transcript shows up
        // here byte-exactly even where the numeric gate has a tolerance.
        // Written the way the library itself reads these fields (`conn->GetStats()->recv_bytes`
        // in cheetah_dot.cc), so this does not depend on whether they are plain integers or atomics.
        auto st = lctx->GetStats();
        fprintf(stderr, "COMM|rank=%d|sent=%llu|recv=%llu\n", rank,
                (unsigned long long)(st->sent_bytes),
                (unsigned long long)(st->recv_bytes));
        fflush(stderr);
    }

    return 0;
}
