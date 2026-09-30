"""Arm the Python census at interpreter start. Bound in as `sitecustomize.py` for the probe only.
Imports snni_pycensus (arms tracemalloc under SNNI_PY_CENSUS_TRACE=1) before the program allocates
anything, which sitecustomize guarantees and a later import would not.
"""
import os

if os.environ.get("SNNI_PY_CENSUS") == "1":
    try:
        import snni_pycensus  # noqa: F401  (importing it is the point)
        # Arm SIGUSR1 here, at interpreter start, the last moment guaranteed to be the main thread
        # (signal.signal only works from it).
        snni_pycensus.arm_signal()
    except Exception:
        pass
