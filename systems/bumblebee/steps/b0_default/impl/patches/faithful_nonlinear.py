#!/usr/bin/env python3
"""faithful_nonlinear: replaces op_bench.cc's placeholder softmax and LayerNorm with the operations
of upstream OpenBumbleBee's Flax BERT (examples/python/ml/flax_bert/flax_bert.py).

Use: off-path diagnostic (placeholder effect on the peak at b0 and b8), never a waterfall step.
- softmax: row maximum (comparison tree), x - max, BumbleBee's neg_exp (exp for x <= 0), row sum,
  reciprocal of the sum broadcast over the row (upstream compiles x / y as x * broadcast(1/y)).
- attention scores scaled by 1/sqrt(d_k) = 1/8 before the softmax.
- LayerNorm: mean, centred input, variance, rsqrt(var + eps), then gamma and beta.
Row reductions halve the column count per level (sum by local adds, max by hal::max).
Usage: faithful_nonlinear.py <op_bench.cc>
"""
import sys

LN_OLD = """    // x shape: (seq_len, hidden_dim), gamma/beta shape: (1, hidden_dim)
    // Broadcast gamma/beta to match x shape
    auto g = hal::broadcast_to(ctx, gamma, x.shape());
    auto b = hal::broadcast_to(ctx, beta, x.shape());
    auto scaled = hal::mul(ctx, x, g);
    auto shifted = hal::add(ctx, scaled, b);
    return shifted;
}"""
LN_NEW = """    // faithful_nonlinear: (x - mean) / sqrt(var + eps) * gamma + beta over the hidden axis.
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
}"""

SM_OLD = """    // Negate for neg_exp
    auto neg_x = hal::negate(ctx, x);
    // BumbleBee neg_exp: computes exp(-x) via Taylor approximation
    auto exp_neg = spu::kernel::hal::intrinsic::nn::f_neg_exp_taylor(
        ctx, neg_x);
    // Sum and reciprocal for normalization
    // For now: element-wise operations capture the key memory cost
    auto recip = hal::reciprocal(ctx, exp_neg);
    auto result = hal::mul(ctx, exp_neg, recip);
    return result;
}"""
SM_NEW = """    // faithful_nonlinear: exp(x - max) / sum(exp(x - max)) per row, as upstream's _softmax.
    auto mx = row_reduce(ctx, x, true);
    auto shifted = hal::sub(ctx, x, hal::broadcast_to(ctx, mx, x.shape()));
    auto nexp = spu::kernel::hal::intrinsic::nn::f_neg_exp_taylor(ctx, shifted);
    auto denom = hal::reciprocal(ctx, row_reduce(ctx, nexp, false));
    return hal::mul(ctx, nexp, hal::broadcast_to(ctx, denom, nexp.shape()));
}"""

HELPER_ANCHOR = "// --------------- Operators ---------------\n"
HELPER = HELPER_ANCHOR + """
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
"""

QK_OLD = "        auto qk = hal::matmul(ctx, qi, kt);\n"
QK_NEW = QK_OLD + """        // faithful_nonlinear: scale the scores by 1/sqrt(d_k) = 1/8 before the softmax.
        qk = hal::mul(ctx, qk, hal::constant(ctx, 0.125F, spu::DT_F32, qk.shape()));
"""


def main():
    path = sys.argv[1]
    t = open(path).read()
    if "faithful_nonlinear" in t:
        sys.exit("FAILED: already patched")
    for name, old in (("layernorm body", LN_OLD), ("softmax body", SM_OLD),
                      ("operators anchor", HELPER_ANCHOR), ("qk matmul", QK_OLD)):
        if t.count(old) != 1:
            sys.exit(f"FAILED: anchor '{name}' matched {t.count(old)} times")
    t = t.replace(HELPER_ANCHOR, HELPER, 1).replace(LN_OLD, LN_NEW, 1)
    t = t.replace(SM_OLD, SM_NEW, 1).replace(QK_OLD, QK_NEW, 1)
    open(path, "w").write(t)
    print("faithful_nonlinear: patched", path)


if __name__ == "__main__":
    main()
