"""Arm the Python census at interpreter start. Bound in as `sitecustomize.py` for the probe only.

It does one thing: import `snni_pycensus`, whose import arms `tracemalloc` when
SNNI_PY_CENSUS_TRACE=1. That has to happen BEFORE the program allocates anything, which is what
`sitecustomize` guarantees and what a later import would not.

The census itself is taken from the MAIN THREAD at named points; see snni_pycensus.py for why the
first version's background thread was withdrawn.
"""
import os

if os.environ.get("SNNI_PY_CENSUS") == "1":
    try:
        import snni_pycensus  # noqa: F401  (importing it is the point)
        # Arm SIGUSR1 here, at interpreter start, because `signal.signal` only works from the
        # main thread and this is the last moment that is guaranteed to be it.
        snni_pycensus.arm_signal()
    except Exception:
        pass
