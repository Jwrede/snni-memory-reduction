// shark_floor_bench.cpp: SHARK BERT-base target check (seq 128, n_embd 768, n_interm 3072), each
// candidate operation run alone. Reports keybytes (KeyBuf.bytesReceived) and peak_kb (VmHWM after a
// clear_refs reset). ONESHOT off (keys streamed from disk).
// Run order: `2` dealer (writes server.dat/client.dat), then `0` server and `1` client.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/ars.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>

using namespace shark;
using namespace shark::protocols;

static long read_status_kb(const char *field) {
    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp) return -1;
    char line[256];
    long v = -1;
    size_t flen = strlen(field);
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, field, flen) == 0) { sscanf(line + flen, "%ld", &v); break; }
    }
    fclose(fp);
    return v;
}

// Reset VmHWM (write 5 to clear_refs) so the next reading is this call's own high point.
static void reset_hwm() {
    FILE *fp = fopen("/proc/self/clear_refs", "w");
    if (fp) { fputs("5\n", fp); fclose(fp); }
}

static unsigned long long key_bytes() {
    if (party == DEALER) return 0;
    return (unsigned long long) dealer->bytesReceived();
}

static unsigned long long k0;
static long rss0;

static void op_begin() {
    reset_hwm();
    rss0 = read_status_kb("VmRSS:");
    k0 = key_bytes();
}

static void op_end(const char *name, unsigned long long n) {
    long hwm = read_status_kb("VmHWM:");
    unsigned long long kb = key_bytes() - k0;
    fprintf(stderr,
            "FLOOR|%s|party=%d|n=%llu|keybytes=%llu|keybytes_per_elem=%.1f|"
            "rss_before_kb=%ld|peak_kb=%ld|delta_kb=%ld\n",
            name, party, n, kb, n ? (double) kb / (double) n : 0.0,
            rss0, hwm, hwm - rss0);
    fflush(stderr);
}

int main(int argc, char **argv) {
    srand(0x5EED);

    // BERT-base, seq 128, as measured by this campaign.
    const u64 n_token  = 128;
    const u64 n_embd   = 768;
    const u64 n_interm = 3072;
    const int f        = 16;

    always_assert(argc > 1);
    int p = atoi(argv[1]);
    const char *oneshot_env = getenv("SHARK_ONESHOT");
    bool oneShot = (oneshot_env && *oneshot_env) ? (atoi(oneshot_env) != 0) : false;

    if (p == DEALER) {
        init::gen(0xdeadbeef);
    } else {
        std::string ip = (argc > 2) ? argv[2] : "127.0.0.1";
        init::eval(p, ip, 42069, oneShot);
    }
    fprintf(stderr, "FLOORCFG|party=%d|oneShot=%d\n", party, (int) oneShot);

    // --- the 768-wide linear output truncation: the batch the earlier campaign pinned at 216 MB
    {
        span<u64> x(n_token * n_embd);
        input::call(x, CLIENT);
        op_begin();
        auto y = ars::call(x, f);
        op_end("ars_linear768", x.size());
    }

    // --- the FFN up-projection's own output truncation, four times wider
    {
        span<u64> x(n_token * n_interm);
        input::call(x, CLIENT);
        op_begin();
        auto y = ars::call(x, f);
        op_end("ars_ffnup3072", x.size());
    }

    // --- GELU's truncation, same width, f+3 rather than f
    {
        span<u64> x(n_token * n_interm);
        input::call(x, CLIENT);
        op_begin();
        auto y = ars::call(x, f + 3);
        op_end("ars_gelu3072", x.size());
    }

    // --- GELU's elementwise square: a Beaver-style multiplication over the same width
    {
        span<u64> x(n_token * n_interm);
        input::call(x, CLIENT);
        op_begin();
        auto y = mul::call(x, x);
        op_end("mul_gelu3072", x.size());
    }

    // --- the heaviest MatMul: FFN up-projection 128 x 768 x 3072
    {
        span<u64> a(n_token * n_embd);
        span<u64> b(n_embd * n_interm);
        input::call(a, CLIENT);
        input::call(b, SERVER);
        op_begin();
        auto z = matmul::call(n_token, n_embd, n_interm, a, b);
        op_end("matmul_ffnup", n_token * n_interm);
    }

    // --- the QKV projection, 128 x 768 x 2304
    {
        span<u64> a(n_token * n_embd);
        span<u64> b(n_embd * n_embd * 3);
        input::call(a, CLIENT);
        input::call(b, SERVER);
        op_begin();
        auto z = matmul::call(n_token, n_embd, n_embd * 3, a, b);
        op_end("matmul_qkv", n_token * n_embd * 3);
    }

    finalize::call();
    fprintf(stderr, "FLOORDONE|party=%d\n", party);
    return 0;
}
