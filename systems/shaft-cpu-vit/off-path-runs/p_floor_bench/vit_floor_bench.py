# ViT SHAFT floor bench (Appendix C parity for shaft-cpu-vit).
# The target is the largest single secret-shared tensor of ViT-Base under SHAFT/CrypTen:
# the feed-forward weight, intermediate_size x hidden_size = 3072 x 768, encoded to int64.
# Reports the resident footprint of that one share, isolated from the float32 source.
import os, gc, torch
def rss_kb():
    for l in open('/proc/self/status'):
        if l.startswith('VmRSS'):
            return int(l.split()[1])
    return -1
import crypten
try:
    crypten.init()
except Exception:
    os.environ.setdefault('WORLD_SIZE','1'); os.environ.setdefault('RANK','0')
    os.environ.setdefault('MASTER_ADDR','127.0.0.1'); os.environ.setdefault('MASTER_PORT','29517')
    crypten.init()
torch.manual_seed(0)
gc.collect(); base = rss_kb()
w = torch.zeros(3072, 768, dtype=torch.float32)    # ViT-Base FFN weight shape
enc = crypten.cryptensor(w)                         # secret-share -> fixed-point int64
del w; gc.collect()                                 # drop the float32 source, isolate the share
try:
    share = enc.share
except AttributeError:
    share = enc._tensor.share
share.add_(1)                                       # touch to force pages resident
after = rss_kb()
nbytes = share.numel() * share.element_size()
print("CRYPTEN", crypten.__version__, "TORCH", torch.__version__)
print("SHARE_DTYPE", share.dtype, "SHAPE", tuple(share.shape))
print("SHARE_BYTES", nbytes, "%.3f MB_decimal" % (nbytes/1e6), nbytes//1024, "KiB")
print("RSS_DELTA_KB", after - base, "(share isolated)")
print("TARGET_KB 18432  (= 3072 * 768 * 8 / 1024)")
