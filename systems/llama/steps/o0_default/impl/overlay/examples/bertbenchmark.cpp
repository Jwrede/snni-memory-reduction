// Authors: Kanav Gupta, Neha Jawalkar
// Copyright:
//
// Copyright (c) 2024 Microsoft Research
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_transformer.h>
#include <sytorch/backend/piranha_cleartext.h>
#include <sytorch/backend/secureml_cleartext.h>
#include <sytorch/backend/float.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>

// D5 (dealer waterfall): weight lazy per-layer alloc flag (see layers.h). Enabled for the dealer.
bool g_lazyWeights = false;
unsigned long long g_weightSeedCtr = 0;
#include <llama/utils.h>
#include <llama/api.h>

template <typename T>
class FFN : public SytorchModule<T>
{
public:
    using SytorchModule<T>::gelu;

    u64 in;
    u64 hidden;

public:
    FC<T> *up;
    FC<T> *down;

    FFN(u64 in, u64 hidden) : in(in), hidden(hidden)
    {
        up = new FC<T>(in, hidden, true);
        down = new FC<T>(hidden, in, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        return down->forward(gelu(up->forward(input)));
    }
};

template <typename T>
class MultiHeadAttention : public SytorchModule<T>
{
public:
    using SytorchModule<T>::split;
    using SytorchModule<T>::view;
    using SytorchModule<T>::add;
    using SytorchModule<T>::transpose;
    using SytorchModule<T>::matmul;
    using SytorchModule<T>::scalarmul;
    using SytorchModule<T>::invsqrt;
    using SytorchModule<T>::softmax;
    using SytorchModule<T>::concat;
    using SytorchModule<T>::attention_mask;

public:
    FC<T> *c_attn;
    FC<T> *c_proj;

    u64 n_heads;
    u64 n_embd;

    MultiHeadAttention(u64 n_heads, u64 n_embd) : n_heads(n_heads), n_embd(n_embd)
    {
        always_assert(n_embd % n_heads == 0);
        c_attn = new FC<T>(n_embd, 3 * n_embd, true);
        c_proj = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &x = c_attn->forward(input);
        auto &qkv_heads = split(x, 3);
        auto &q_heads = view(qkv_heads, 0);
        auto &k_heads = view(qkv_heads, 1);
        auto &v_heads = view(qkv_heads, 2);
        auto &qs = split(q_heads, n_heads);
        auto &ks = split(k_heads, n_heads);
        auto &vs = split(v_heads, n_heads);

        double divisor = 1 / sqrt(double(n_embd) / double(n_heads));

        std::vector<Tensor<T> *> qks_sm_vs;
        for (u64 i = 0; i < n_heads; ++i)
        {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);
            auto &kt = transpose(k);
            auto &qk = matmul(q, kt);
            auto &qks = scalarmul(qk, divisor);

            auto &qks_sm = softmax(qks);

            auto &qks_sm_v = matmul(qks_sm, v);
            qks_sm_vs.push_back(&qks_sm_v);
        }

        auto &qks_sm_vs_cat = concat(qks_sm_vs);
        auto &res = c_proj->forward(qks_sm_vs_cat);
        return res;
    }
};

template <typename T>
class TransformerBlock : public SytorchModule<T>
{
public:
    using SytorchModule<T>::add;

    MultiHeadAttention<T> *attn;
    FFN<T> *ffn;
    LayerNorm<T> *ln0;
    LayerNorm<T> *ln1;

    u64 n_heads, n_embd;

public:
    TransformerBlock(u64 n_heads, u64 n_embd) : n_heads(n_heads), n_embd(n_embd)
    {
        attn = new MultiHeadAttention<T>(n_heads, n_embd);
        ffn = new FFN<T>(n_embd, 4 * n_embd);
        ln0 = new LayerNorm<T>(n_embd);
        ln1 = new LayerNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &attn_out = attn->forward(input);
        auto &add0_out = add(attn_out, input);
        auto &ln0_out = ln0->forward(add0_out);

        auto &ffn_out = ffn->forward(ln0_out);
        auto &add1_out = add(ffn_out, ln0_out);
        auto &ln1_out = ln1->forward(add1_out);
        return ln1_out;
    }
};

template <typename T>
class BERT : public SytorchModule<T>
{
public:
    using SytorchModule<T>::tanh;
    using SytorchModule<T>::view;
    using SytorchModule<T>::add;
    using SytorchModule<T>::unsqueeze;
    std::vector<TransformerBlock<T> *> blocks;
    LayerNorm<T> *ln_f;
    FC<T> *pool;
    u64 n_layer, n_heads, n_embd;

public:
    BERT(u64 n_layer, u64 n_heads, u64 n_embd) : n_layer(n_layer), n_heads(n_heads), n_embd(n_embd)
    {
        for (u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlock<T>(n_heads, n_embd));
        }
        ln_f = new LayerNorm<T>(n_embd);
        pool = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &y = ln_f->forward(input);
        Tensor<T> *x = &y;
        // Tensor<T> *x = &input;

        for (u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }

        return *x;
    }
};

int main(int __argc, char **__argv)
{
    sytorch_init();

    // bert tiny
    const u64 n_embd = 768;
    const u64 n_head = n_embd / 64;
    const u64 n_layer = 12;
    const u64 scale = 12;
    const u64 bw = 51;
    const u64 n_seq = 128;

    // bert base
    // const u64 n_embd = 768;
    // const u64 n_head = 12;
    // const u64 n_layer = 12;
    // const u64 scale = 12;
    // const u64 bw = 51;
    // const u64 n_seq = 128;

    // bert large
    // const u64 n_embd = 1024;
    // const u64 n_head = n_embd / 64;
    // const u64 n_layer = 24;
    // const u64 scale = 12;
    // const u64 bw = 51;
    // const u64 n_seq = 128;

    int party = atoi(__argv[1]);
    std::string ip = "127.0.0.1";
    if (__argc > 2)
        ip = __argv[2];

    using LlamaVersion = LlamaTransformer<u64>;
    LlamaVersion *llama = new LlamaVersion();
    // PINNED, not clock-seeded. A gate compares each step against the BASELINE's output, and
    // that comparison is meaningless unless the system produces the same output twice. The
    // key PRNG (LlamaConfig::prngs) is separately pinned already, which is why the dealer's
    // key material reproduces; this closes the remaining clock dependency.
    srand(0x5EEDU);
    fprintf(stderr, "SEEDED|srand=0x5eed|party=%d\n", party);

    LlamaConfig::bitlength = bw;
    LlamaConfig::party = party;

    // D5 (dealer waterfall): only the dealer runs weight lazy-alloc (it never loads real weights;
    // net.load() is guarded by party != DEALER below, so dummy weights are pure S1 waste here).
    g_lazyWeights = (party == DEALER);

    llama->init(ip, true);

    BERT<u64> net(n_layer, n_head, n_embd);
    Tensor<u64> input({n_seq, n_embd});
    net.init(scale, input);
    net.setBackend(llama);
    net.optimize();
    if (party != DEALER)
    {
        // THE ONLINE PARTIES USE THE SAME DUMMY WEIGHTS THE DEALER ALREADY GENERATED, and that is
        // a correctness requirement rather than a convenience. The dealer masks the weights it
        // holds; if the online parties held different ones the protocol would be inconsistent.
        //
        // Since d5, every FC layer draws its weights from its OWN deterministically seeded stream
        // in `_initScale`, applied unconditionally to all parties. So all three parties already
        // hold byte-identical weights at this point, by construction, and loading is not needed.
        //
        // The stock calls cannot run in any case: `bert-tiny-weights.dat` and `15469.dat` are not
        // in the image, are not downloaded, and the first is bert-tiny sized while the dimensions
        // above are bert-base. This path has never been exercised at these dimensions.
        //
        // The input is filled deterministically for the same reason the seed is pinned: the gate
        // compares this run's output against the baseline's, and an unset input would make that
        // comparison meaningless. Values are kept small so that twelve layers of accumulation
        // cannot wrap the 51-bit ring, which would make the observable depend on ring width
        // rather than on the computation.
        for (u64 i = 0; i < input.size(); ++i)
        {
            i64 v = (i64)((i * 2654435761ULL) % 2001) - 1000;
            // den=1000 gives +-1.0, since 1.0 is (1<<scale). THE FIRST VERSION DIVIDED BY
            // 1000000 AND PRODUCED A GATE THAT COULD NOT FAIL: that yields values in [-4, 4]
            // against a 1.0 of 4096, so every input was numerically zero and the reconstructed
            // output came out as 98,304 exact zeros -- measured 2026-08-04 in
            // `p_iotrace`. It would have reproduced perfectly across runs, so the determinism
            // check would have PASSED on an observable with no sensitivity to the computation.
            // This is the same defect SIGMA-GPU's patch already records and corrects, and this
            // file inherited it. The weights are unaffected: FC::_initScale randomizes them at
            // xavier * (1<<scale) ~ 148, which is not near zero.
            input.data[i] = (u64)((v * (i64)(1ULL << scale)) / 1000);
        }
    }

    llama->initializeInferencePartyA(net.root);
    llama->initializeInferencePartyB(input);

    llama::start();
    net.forward(input);
    llama::end();

    auto &output = net.activation;
    llama->outputA(output);
    llama->finalize();

    // THE GATE OBSERVABLE IS STRUCTURAL ON THIS SYSTEM (tier T1s). The value observable below is
    // kept as evidence but is degenerate: 98,304 exact zeros, for the reason p_reveal measured.
    // So what a step is compared on is the protocol traffic, accumulated in comms.cpp.
    //
    // BOTH online parties emit it. They receive different messages, so the two lines differ from
    // each other by design; what must reproduce is each line across runs of the same program.
    if (party != DEALER)
    {
        extern uint64_t g_snni_wire_h, g_snni_wire_sum, g_snni_wire_ops, g_snni_wire_bytes;
        fprintf(stderr,
                "GATE|wire|party=%d|ops=%lu|bytes=%lu|ordered=%016lx|commutative=%016lx\n",
                party, (unsigned long)g_snni_wire_ops, (unsigned long)g_snni_wire_bytes,
                (unsigned long)g_snni_wire_h, (unsigned long)g_snni_wire_sum);
    }

    if (party == CLIENT)
    {
        auto signedAct = Tensor<i64>((i64*) net.activation.data, net.activation.shape);
        print(signedAct, scale, bw);

        // THE GATE OBSERVABLE: the whole reconstructed output, as raw ring elements, printed
        // exactly. Not an argmax: a label is one bit and would pass two runs whose values differed
        // materially, which is what this gate exists to catch. Only the CLIENT reconstructs.
        // THE GATE MUST BE SENSITIVE, NOT MERELY PRESENT. An all-zero output reproduces
        // perfectly and distinguishes nothing, which is how a check that cannot fail looks from
        // the outside. Counted here and announced, so the harness can refuse the run.
        {
            unsigned long nz = 0;
            for (u64 i = 0; i < net.activation.size(); ++i)
                if (((i64 *)net.activation.data)[i] != 0) ++nz;
            fprintf(stderr, "GATE_NONZERO|%lu/%lu\n", nz, (unsigned long)net.activation.size());
            if (nz == 0)
                fprintf(stderr, "GATE_DEGENERATE|every observable value is zero\n");
        }
        fprintf(stderr, "GATE|logits|n=%lu|", (unsigned long)net.activation.size());
        for (u64 i = 0; i < net.activation.size(); ++i)
            fprintf(stderr, "%ld%s", (long)((i64 *)net.activation.data)[i],
                    i + 1 < net.activation.size() ? "," : "");
        fprintf(stderr, "\n");
    }
    return 0;
}