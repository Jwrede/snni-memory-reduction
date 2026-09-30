// Instrumented BERT-base benchmark for SHARK per-operator memory profiling
// (upstream benchmarks/bert.cpp plus MEM markers at operator boundaries).
//
// Environment:
//   SHARK_SEQ_LEN    input sequence length (default 32)
//   SHARK_N_LAYERS   number of transformer layers (default 12)
//
// Markers (stderr): MEM|<party>|<layer>|<phase>|<op>|<rss_kb>|<timestamp_ms>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>

#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/relu.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/conv.hpp>
#include <shark/protocols/add.hpp>
#include <shark/protocols/ars.hpp>
#include <shark/protocols/reciprocal.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/common.hpp>

#include <shark/utils/timer.hpp>
#include <shark/utils/assert.hpp>

using namespace shark;
using namespace shark::protocols;

int f = 16;

// --------------- Memory instrumentation helpers ---------------

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

static void mem(const char* phase, const char* op, int layer_idx) {
    long rss = read_vmrss_kb();
    long long ts = now_ms();
    fprintf(stderr, "MEM|%d|%d|%s|%s|%ld|%lld\n", party, layer_idx, phase, op, rss, ts);
    fflush(stderr);
}

// --------------- Deterministic non-zero input (campaign setup, not a lever) ---------------
//
// WHY THIS IS HERE, and it was found by measurement rather than by reading. `shark::span<T>(n)` is
// `new T[n]`, and neither the input vector nor any weight is ever written before `input::call`
// adds the dealer's mask to it. Large spans come from mmap, so they are zero, and the first
// determinism check of this system produced a reconstructed output of 98,304 elements that were
// ALL EXACTLY ZERO, byte-identical across two runs and across both parties.
//
// That is a gate that cannot fail. It proves the run completed and nothing else, which is exactly
// the defect this campaign found on SIGMA-GPU (`input.zero()` there, uninitialised memory here)
// and the reason PROCEDURE.md insists the observable be something a step could plausibly change.
//
// So the input and the weights are filled from a fixed arithmetic sequence. The magnitudes are
// chosen so that twelve layers neither vanish nor wrap:
//
//   f = 16, so 1.0 is 65536, and the input is drawn from [0, 1).
//   Every linear sums `b` products and truncates by f, so a weight near 2^16 / b keeps the
//   magnitude constant across the layer. That is ~85 for the three 768-wide linears and ~21 for
//   the 3072-wide FFN down-projection.
//   The quadratic steps (the QK matmul and GELU's square) shrink anything below 1.0 rather than
//   growing it, and GELU's additive `1 << (2f+2)` term keeps the result off zero.
//
// It is setup, not a lever: identical for the baseline and for every step, it changes no
// allocation and no size, and FSS key material is data-independent so it cannot move the peak.
// Fixed inputs exist so the gate can exist; they are not a deployment configuration.
static inline u64 fill_stream(u64 idx, u64 salt) {
    u64 z = idx + 0x9E3779B97F4A7C15ull * (salt + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void fill_span(span<u64> &s, u64 salt, u64 modulus) {
    for (u64 i = 0; i < s.size(); ++i) s[i] = fill_stream(i, salt) % modulus;
}

// --------------- Correctness gate (campaign setup, not a lever) ---------------
//
// The stock benchmark reconstructs its output and then throws it away: nothing is printed, so
// there is no observable a step could be compared against. Every waterfall step must be shown to
// leave the computation unchanged, and that comparison is against the BASELINE's own output.
//
// Emitted here rather than computed afterwards, because only the parties themselves hold the
// reconstructed vector. Both online parties reconstruct the same value, so both emit; the party
// field is part of the line so a per-party difference would be visible rather than averaged away.
//
// Two forms, for two different jobs:
//   GATE|out|...   one line, a 64-bit FNV-1a over the whole vector, which is what make_steps.py
//                  compares (its tokeniser takes one token per line).
//   GATE|out<i>|.. the first eight values verbatim, so a mismatch carries a magnitude and not
//                  only a verdict.
// The FULL vector is written to $SHARK_OUT_DUMP when that is set, so the size of any future
// difference can be computed from the archived runs rather than re-run.
static void emit_gate(span<u64> &y) {
    unsigned long long h = 14695981039346656037ULL;
    const unsigned char* p = (const unsigned char*) y.data();
    size_t nbytes = (size_t) y.size() * sizeof(u64);
    for (size_t i = 0; i < nbytes; ++i) { h ^= (unsigned long long) p[i]; h *= 1099511628211ULL; }
    fprintf(stderr, "GATE|out|party=%d|n=%lu|fnv1a64=%016llx\n",
            party, (unsigned long) y.size(), h);

    // A CONSTANT OUTPUT IS A GATE THAT CANNOT FAIL, so the run says whether it produced one. This
    // system's first determinism check reproduced perfectly with an output of 98,304 zeros, which
    // proved only that the protocol had run. The harness reads this line and fails the run if the
    // output is constant, so the defect can never come back silently.
    // Deliberately NOT prefixed `GATE|`: it is provenance about the observable, not the observable.
    unsigned long long nonzero = 0, differing = 0;
    for (u64 i = 0; i < y.size(); ++i) {
        if (y[i] != 0) nonzero++;
        if (y[i] != y[0]) differing++;
    }
    fprintf(stderr, "GATEQ|party=%d|n=%lu|nonzero=%llu|differing_from_first=%llu\n",
            party, (unsigned long) y.size(), nonzero, differing);
    for (u64 i = 0; i < 8 && i < y.size(); ++i)
        fprintf(stderr, "GATE|out%llu|party=%d|%llu\n",
                (unsigned long long) i, party, (unsigned long long) y[i]);
    fflush(stderr);

    const char* dump = getenv("SHARK_OUT_DUMP");
    if (dump && *dump) {
        FILE* fp = fopen(dump, "w");
        if (fp) {
            for (u64 i = 0; i < y.size(); ++i)
                fprintf(fp, "%llu\n", (unsigned long long) y[i]);
            fclose(fp);
        } else {
            fprintf(stderr, "GATEDUMP|FAILED|%s\n", dump);
        }
    }
}

// --------------- Model ---------------

struct BertModel {
    u64 n_layers;
    u64 n_heads = 12;
    u64 n_embd = 768;
    u64 n_interm = 3072;
    u64 logdivisor = 3;

    std::vector<span<u64>> c_attn_w;
    std::vector<span<u64>> c_attn_b;
    std::vector<span<u64>> c_proj_w;
    std::vector<span<u64>> c_proj_b;
    std::vector<span<u64>> ffn_up_w;
    std::vector<span<u64>> ffn_up_b;
    std::vector<span<u64>> ffn_down_w;
    std::vector<span<u64>> ffn_down_b;

    BertModel(u64 layers) : n_layers(layers),
        c_attn_w(layers), c_attn_b(layers),
        c_proj_w(layers), c_proj_b(layers),
        ffn_up_w(layers), ffn_up_b(layers),
        ffn_down_w(layers), ffn_down_b(layers)
    {
        for (u64 i = 0; i < n_layers; ++i) {
            c_attn_w[i] = span<u64>(n_embd * 3 * n_embd);
            c_attn_b[i] = span<u64>(n_embd * 3);
            c_proj_w[i] = span<u64>(n_embd * n_embd);
            c_proj_b[i] = span<u64>(n_embd);
            ffn_up_w[i] = span<u64>(n_embd * n_interm);
            ffn_up_b[i] = span<u64>(n_interm);
            ffn_down_w[i] = span<u64>(n_interm * n_embd);
            ffn_down_b[i] = span<u64>(n_embd);
        }
    }
};

// --------------- Operators ---------------

span<u64> gelu(span<u64> &x, int layer_idx) {
    auto x2 = mul::call(x, x);
    if (layer_idx == 0) mem("gelu", "after_mul", layer_idx);

    span<u64> y(x.size());
    for (u64 i = 0; i < x.size(); ++i) {
        y[i] = x2[i] + x[i] * (1ull << (f + 1));
        if (party != DEALER) {
            y[i] += (1ull << (2*f + 2));
        }
    }
    auto result = ars::call(y, f + 3);
    if (layer_idx == 0) mem("gelu", "after_ars", layer_idx);
    return result;
}

span<u64> softmax(u64 a, u64 b, span<u64> &x, int layer_idx) {
    always_assert(x.size() == (a * b));
    auto exps = relu::call(x);
    if (layer_idx == 0) mem("softmax", "after_relu", layer_idx);

    span<u64> den(a);
    for (u64 i = 0; i < a; ++i) {
        u64 sum = 0;
        for (u64 j = 0; j < b; ++j) {
            sum += exps[i * b + j];
        }
        den[i] = sum + 1 * (party != DEALER);
    }

    auto den_inv = reciprocal::call(den, f);
    if (layer_idx == 0) mem("softmax", "after_reciprocal", layer_idx);

    span<u64> den_inv_expanded(a * b);
    for (u64 i = 0; i < a; ++i) {
        for (u64 j = 0; j < b; ++j) {
            den_inv_expanded[i * b + j] = den_inv[i];
        }
    }

    auto y = mul::call(exps, den_inv_expanded);
    if (layer_idx == 0) mem("softmax", "after_mul", layer_idx);
    return lrs::call(y, f);
}

span<u64> linear(u64 a, u64 b, u64 c, span<u64> &x, span<u64> &w, span<u64> &bias) {
    auto u = matmul::call(a, b, c, x, w);
    auto v = add::call(u, bias);
    auto y = ars::call(v, f);
    return y;
}

// --------------- View / Transpose / Concat ---------------

span<u64> view(span<u64> &x, int n_token, int n_pieces, int idx) {
    always_assert(x.size() % (n_token * n_pieces) == 0);
    int n_embd = x.size() / n_token;
    span<u64> y(x.size() / n_pieces);
    for (int i = 0; i < n_token; i++) {
        for (int j = 0; j < (n_embd / n_pieces); j++) {
            y[i * (n_embd / n_pieces) + j] = x[i * n_embd + j + idx * (n_embd / n_pieces)];
        }
    }
    return y;
}

span<u64> transpose(u64 a, u64 b, span<u64> &x) {
    always_assert(x.size() == (a * b));
    span<u64> y(x.size());
    for (int i = 0; i < (int)a; i++) {
        for (int j = 0; j < (int)b; j++) {
            y[j * a + i] = x[i * b + j];
        }
    }
    return y;
}

void concat(span<u64> &x, span<u64> &y, int n_token, int idx) {
    int x_n_embd = x.size() / n_token;
    int y_n_embd = y.size() / n_token;
    for (int i = 0; i < n_token; i++) {
        for (int j = 0; j < y_n_embd; j++) {
            x[i * x_n_embd + j + idx * y_n_embd] = y[i * y_n_embd + j];
        }
    }
}

// --------------- MHA ---------------

span<u64> mha(span<u64> &x, int layer_idx, BertModel &model) {
    u64 n_token = x.size() / model.n_embd;
    mem("mha", "begin", layer_idx);

    auto c = linear(n_token, model.n_embd, model.n_embd * 3, x, model.c_attn_w[layer_idx], model.c_attn_b[layer_idx]);
    mem("mha", "after_qkv_linear", layer_idx);

    auto q = view(c, n_token, 3, 0);
    auto k = view(c, n_token, 3, 1);
    auto v = view(c, n_token, 3, 2);

    double logdivisor = model.logdivisor;

    span<u64> qks_sm_vs(model.n_embd * n_token);
    for (int i = 0; i < (int)model.n_heads; i++) {
        auto qi = view(q, n_token, model.n_heads, i);
        auto ki = view(k, n_token, model.n_heads, i);
        auto vi = view(v, n_token, model.n_heads, i);

        auto kt = transpose(n_token, model.n_embd / model.n_heads, ki);
        auto qk = matmul::call(n_token, model.n_embd / model.n_heads, n_token, qi, kt);
        auto s = ars::call(qk, f + logdivisor);

        // Sub-operator detail only for head 0
        if (i == 0) mem("mha", "after_qk_matmul", layer_idx);

        auto qks_sm = softmax(n_token, n_token, s, (i == 0) ? layer_idx : -1);
        if (i == 0) mem("mha", "after_softmax", layer_idx);

        auto qks_sm_v = matmul::call(n_token, n_token, model.n_embd / model.n_heads, qks_sm, vi);
        if (i == 0) mem("mha", "after_av_matmul", layer_idx);

        concat(qks_sm_vs, qks_sm_v, n_token, i);
    }
    mem("mha", "after_all_heads", layer_idx);

    auto z = ars::call(qks_sm_vs, f);
    auto proj = linear(n_token, model.n_embd, model.n_embd, z, model.c_proj_w[layer_idx], model.c_proj_b[layer_idx]);
    mem("mha", "after_proj_linear", layer_idx);

    return proj;
}

// --------------- FFN ---------------

span<u64> ffn(span<u64> &x, int layer_idx, BertModel &model) {
    u64 n_token = x.size() / model.n_embd;
    mem("ffn", "begin", layer_idx);

    auto c = linear(n_token, model.n_embd, model.n_interm, x, model.ffn_up_w[layer_idx], model.ffn_up_b[layer_idx]);
    mem("ffn", "after_up_linear", layer_idx);

    auto d = gelu(c, layer_idx);
    mem("ffn", "after_gelu", layer_idx);

    auto g = linear(n_token, model.n_interm, model.n_embd, d, model.ffn_down_w[layer_idx], model.ffn_down_b[layer_idx]);
    mem("ffn", "after_down_linear", layer_idx);

    return g;
}

// --------------- LayerNorm (passthrough in SHARK) ---------------

span<u64> layernorm(span<u64> &x, int layer_idx, BertModel &model) {
    return x;
}

// --------------- Layer / Inference ---------------

span<u64> bert_layer(span<u64> &x, int layer_idx, BertModel &model) {
    mem("layer", "begin", layer_idx);

    auto a = layernorm(x, layer_idx, model);
    auto b = mha(a, layer_idx, model);
    auto c = add::call(a, b);
    mem("layer", "after_mha_residual", layer_idx);

    auto d = layernorm(c, layer_idx, model);
    auto e = ffn(d, layer_idx, model);
    auto g = add::call(d, e);
    mem("layer", "after_ffn_residual", layer_idx);

    // ================== s1_batch_check_stream: THE LEVER ==================
    //
    // CHOSEN FROM THE MEASURED DECOMPOSITION OF s0's PEAK (58,095,268 kB). The largest object there
    // is the 55 GB preprocessing slurp, and its only fix is a reload that MEASURABLY costs runtime
    // (three mechanisms measured, +26.9% / +6.4% / see off-path.md), so it is a paid fix and the
    // rule takes the largest object with a FREE one instead. That is
    // `std::_Vector_base<unsigned __int128>::_M_allocate` at 720.7 MB, which is
    // `batchCheckArithmBuffer` (common.cpp:22), together with its u64 twin
    // `batchCheckBoolBuffer` at 360.6 MB.
    //
    // WHY IT IS DEAD MEMORY. Every `authenticated_reconstruct` pushes one MAC residual per
    // reconstructed element into those two vectors (common.cpp:309, 370) and NOTHING ever consumes
    // them until `batch_check()` runs -- which happens only inside `output::eval`, i.e. once, after
    // the last layer. So the residuals of layer 0 sit in memory for the whole run while being
    // needed for a single multiply-accumulate. Retained past last use, which is the procedure's
    // definition of a dead object.
    //
    // THE FIX. Run the check at the end of every layer. `batch_check()` is already public
    // (common.hpp:68), so this is a call in the measured source and no library patch.
    // `std::vector::clear()` keeps capacity, so the buffers settle at ONE layer's worth instead of
    // twelve. The dealer never reconstructs, so its buffers are empty and `batch_check()` would
    // return immediately, but it is guarded anyway because the dealer has no `peer`.
    //
    // IT IS NOT A SECURITY TRADE, and that is worth arguing rather than asserting. The check
    // verifies a random linear combination of MAC residuals against the peer's. Splitting one batch
    // into twelve verifies every residual exactly as before, with an independently committed seed
    // per batch; the soundness error goes from ~2^-64 to ~12 * 2^-64, and an adversary now has to
    // pass twelve checks instead of one and is caught EARLIER rather than later. Nothing is dropped
    // from the check. What it costs is 11 extra commit-and-open exchanges of a hash and a 128-bit
    // value over loopback.
    //
    // On the evaluators `prngGlobal` feeds commitment randomness only -- `randomize()` is the
    // dealer's path -- so advancing it earlier cannot move the protocol output. The gate is what
    // establishes that rather than this comment.
    if (party != DEALER) {
        batch_check();
        if (layer_idx == 0) mem("layer", "after_batch_check", layer_idx);
    }

    return g;
}

span<u64> inference(span<u64> &x, BertModel &model) {
    span<u64> y = x;
    for (u64 i = 0; i < model.n_layers; i++) {
        y = bert_layer(y, (int)i, model);
    }
    return y;
}

// --------------- Input sharing ---------------

void model_input(BertModel &model) {
    // A weight near 2^16/b keeps a b-wide linear magnitude-preserving; biases share the input's
    // scale. The salt makes every tensor of every layer a different sequence while staying a fixed
    // function of (layer, tensor), so the weights are reproducible without any generator state.
    const u64 W768 = 171;    // 2 * (65536 / 768),  mean ~85
    const u64 W3072 = 43;    // 2 * (65536 / 3072), mean ~21
    for (u64 i = 0; i < model.n_layers; i++) {
        fill_span(model.c_attn_w[i],  8 * i + 0, W768);
        fill_span(model.c_attn_b[i],  8 * i + 1, 65536);
        fill_span(model.c_proj_w[i],  8 * i + 2, W768);
        fill_span(model.c_proj_b[i],  8 * i + 3, 65536);
        fill_span(model.ffn_up_w[i],  8 * i + 4, W768);
        fill_span(model.ffn_up_b[i],  8 * i + 5, 65536);
        fill_span(model.ffn_down_w[i], 8 * i + 6, W3072);
        fill_span(model.ffn_down_b[i], 8 * i + 7, 65536);

        input::call(model.c_attn_w[i], SERVER);
        input::call(model.c_attn_b[i], SERVER);
        input::call(model.c_proj_w[i], SERVER);
        input::call(model.c_proj_b[i], SERVER);
        input::call(model.ffn_up_w[i], SERVER);
        input::call(model.ffn_up_b[i], SERVER);
        input::call(model.ffn_down_w[i], SERVER);
        input::call(model.ffn_down_b[i], SERVER);
    }
}

// --------------- Main ---------------

int main(int argc, char **argv) {
    // PIN THE C RNG BEFORE init (campaign setup, not a lever).
    //
    // The dealer's own generator is already pinned in the library: init::gen seeds prngGlobal with
    // the constant 0xdeadbeef. The ONLINE parties are seeded differently, from
    // `toBlock(::rand(), ::rand())` (init.cpp:53), and nothing in this program or the library ever
    // calls srand. glibc then behaves as if srand(1) had been called, so the online seed is a
    // constant BY ACCIDENT rather than by declaration. Pinning it explicitly makes the seed a
    // stated property of the measured system instead of a libc default that a future toolchain or
    // a future line of code could move without any symptom.
    //
    // Fixed randomness exists so the gate can exist. It is applied identically to the baseline and
    // to every step, exactly as the thread count is pinned. It is NOT how the protocol would be
    // deployed: fixed shares would be a security defect in a real deployment.
    srand(0x5EED);

    init::from_args(argc, argv);
    fprintf(stderr, "LEVER|s1_batch_check_stream|per_layer=1|party=%d\n", party);
    fprintf(stderr, "SEEDED|srand=0x5eed|dealer_prng=0xdeadbeef|party=%d\n", party);
    fflush(stderr);

    const char* layers_env = getenv("SHARK_N_LAYERS");
    u64 n_layers = layers_env ? (u64)std::atoi(layers_env) : 12;

    const char* seq_env = getenv("SHARK_SEQ_LEN");
    u64 n_token = seq_env ? (u64)std::atoi(seq_env) : 32;

    fprintf(stderr, "CONFIG|party=%d|n_layers=%lu|seq_len=%lu\n", party, n_layers, n_token);

    mem("main", "after_init", -1);

    BertModel model(n_layers);
    mem("main", "after_model_alloc", -1);

    span<u64> x(n_token * model.n_embd);
    mem("main", "after_input_alloc", -1);

    // The activation entering layer 0, in [0, 1) at f = 16. Filled on every party: the owner's
    // content IS the secret input (`input::call` adds the dealer's mask to whatever is there),
    // and on the other parties the buffer is overwritten by the receive, so filling it costs
    // nothing and removes any dependence on what `new u64[]` happened to hand back.
    fill_span(x, 0xA11CE, 65536);

    utils::start_timer("input");
    input::call(x, CLIENT);
    mem("main", "after_client_input", -1);

    model_input(model);
    mem("main", "after_model_input", -1);
    utils::stop_timer("input");

    if (party != DEALER)
        peer->sync();
    mem("main", "after_sync", -1);

    utils::start_timer("bert");
    auto y = inference(x, model);
    output::call(y);
    mem("main", "after_output", -1);
    if (party != DEALER) emit_gate(y);
    utils::stop_timer("bert");

    finalize::call();
    mem("main", "after_finalize", -1);

    utils::print_all_timers();

    return 0;
}
