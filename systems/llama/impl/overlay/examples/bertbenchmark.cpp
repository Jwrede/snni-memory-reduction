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
    // Pinned, not clock-seeded, so the output reproduces across runs (the key PRNG is pinned
    // separately); this closes the remaining clock dependency.
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
        // All three parties already hold byte-identical dummy weights (since d5 every FC layer
        // draws from its own deterministically seeded _initScale stream), so net.load() is not
        // needed and its stock files are absent anyway. The input is filled deterministically and
        // kept small so 12 layers cannot wrap the 51-bit ring.
        for (u64 i = 0; i < input.size(); ++i)
        {
            i64 v = (i64)((i * 2654435761ULL) % 2001) - 1000;
            // den=1000 gives +-1.0 (1.0 is 1<<scale). Dividing by 1000000 instead put every input
            // in [-4,4] i.e. numerically zero, giving a degenerate all-zero gate.
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

    // The gate observable is structural (T1s): the value observable below is degenerate (all zeros),
    // so steps are compared on the protocol traffic accumulated in comms.cpp. Both online parties
    // emit it; the two lines differ by design, and each must reproduce across runs.
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

        // Gate observable: the whole reconstructed output as raw ring elements, printed exactly
        // (not an argmax). Only the CLIENT reconstructs. Nonzero-counted and announced so the
        // harness can refuse a degenerate all-zero output.
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