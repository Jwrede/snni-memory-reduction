# SHAFT BERT target micro-benchmark (Appendix B.2, Section 4.5.5).
# Builds the terms of the shaft-cpu target with CrypTen's own objects, one at a time, and reads
# the resident footprint of each from /proc/self/status:
#   table share      28,996 x 768 secret-shared int64      178,151,424 B
#   one-hot matrix   128 x 28,996 secret-shared int64        29,691,904 B
#   slice triple     Beaver triple of one of 16 slices        13,782,016 B
#                    (128x1813, 1813x768, 128x768), drawn from CrypTen's provider
#   output + sum     2 x 128 x 768 secret-shared int64        1,572,864 B
#   target                                                  223,198,208 B
# After each term the float/plaintext sources are dropped, gc runs and malloc_trim(0) returns
# glibc's freed pages, so a delta counts the live shares and not allocator retention.
import os, gc, ctypes, torch

V, D, S, N = 28996, 768, 128, 16
v = (V + N - 1) // N  # 1,813 rows per slice, as s7 computes chunk_size

libc = ctypes.CDLL("libc.so.6")

def settle():
    gc.collect()
    libc.malloc_trim(0)

def rss_kb():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"):
            return int(l.split()[1])
    return -1

def share_of(x):
    t = x._tensor if hasattr(x, "_tensor") else x
    return t.share

import crypten
os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29518")
crypten.init()
torch.manual_seed(0)
provider = crypten.mpc.get_default_provider()
print("CRYPTEN", crypten.__version__, "TORCH", torch.__version__, "PROVIDER", type(provider).__name__)

settle(); base = rss_kb(); prev = base
rows = []

def record(name, derived_b, tensors):
    global prev
    for t in tensors:
        t.add_(1)  # touch every page
    settle()
    now = rss_kb()
    nbytes = sum(t.numel() * t.element_size() for t in tensors)
    rows.append((name, derived_b, nbytes, now - prev))
    print(f"FLOOR|{name}|tensor_bytes={nbytes}|derived_bytes={derived_b}|rss_delta_kb={now - prev}|vmrss_kb={now}")
    prev = now

src = torch.zeros(V, D); table = crypten.cryptensor(src); del src
record("table_share", V * D * 8, [share_of(table)])

idx = torch.randint(0, V, (S,))
src = torch.nn.functional.one_hot(idx, V).float(); onehot = crypten.cryptensor(src); del src
record("one_hot", S * V * 8, [share_of(onehot)])

a, b, c = provider.generate_additive_triple((S, v), (v, D), "matmul")
record("slice_triple", (S * v + v * D + S * D) * 8, [a.share, b.share, c.share])

src = torch.zeros(S, D); out = crypten.cryptensor(src); acc = crypten.cryptensor(src); del src
record("output_and_sum", 2 * S * D * 8, [share_of(out), share_of(acc)])

tot_derived = sum(r[1] for r in rows); tot_bytes = sum(r[2] for r in rows); tot_kb = prev - base
print(f"FLOOR_SUMMARY|derived_bytes={tot_derived}|tensor_bytes={tot_bytes}|rss_delta_kb={tot_kb}"
      f"|rss_delta_bytes={tot_kb * 1024}|ratio={tot_kb * 1024 / tot_derived:.4f}")
