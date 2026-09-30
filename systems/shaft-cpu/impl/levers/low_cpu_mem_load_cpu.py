"""Checkpoint loaded without materialising it twice (low_cpu_mem_usage=True).
Object (p_census_peak_x2, rss 1,121,180 vs 1,150,240 kB peak, 97.5%): 866.5 MB float32 weights in pairs
(49 flat vs 49 shaped; deduplicated by data_ptr): from_pretrained holds state_dict and module. Markers:
after_init 249,388 kB; after_model_load 1,150,172 (peak, 0.1 s later); after_extract_submodules 384,688.
Fix: meta-device build filled tensor by tensor (load high-water ~900 MB -> 21 MB). Donor: retired
08_low_cpu_mem_load (in-process half only; its shard split is outside the process).
Check: mmap-filled weights may become file-backed (file:, excluded from W): a peak fall matched by a
file: rise is bookkeeping. Free; gate expected identical.
"""

import sys

import transformers
from transformers import AutoModelForSequenceClassification as _Auto

_ORIG = _Auto.from_pretrained.__func__ if hasattr(_Auto.from_pretrained, "__func__") \
    else _Auto.from_pretrained


def from_pretrained(cls, *args, **kwargs):
    # setdefault: an explicit caller argument wins over this lever's default.
    kwargs.setdefault("low_cpu_mem_usage", True)
    return _ORIG(cls, *args, **kwargs)


_Auto.from_pretrained = classmethod(from_pretrained)

# Without accelerate the keyword raises instead of being ignored; fail at import time with a reason.
try:
    import accelerate  # noqa: F401
except ImportError:
    print("FATAL|low_cpu_mem_load_cpu|accelerate is not installed; low_cpu_mem_usage=True would "
          "raise inside from_pretrained", file=sys.stderr, flush=True)
    raise

print(f"LEVER|low_cpu_mem_load_cpu|patched|transformers={transformers.__version__}|"
      f"accelerate={accelerate.__version__}", file=sys.stderr, flush=True)
