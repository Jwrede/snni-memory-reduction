#!/usr/bin/env python3
"""Write a recording-pool Apptainer def derived from a PUBLISHED step image, so the instrument binary differs from the measured one by the recorder alone.

    make_prm_def.py <parent step id> <out.def>
"""
import io
import sys

if len(sys.argv) != 3:
    sys.exit(__doc__)
PARENT, OUT = sys.argv[1], sys.argv[2]
B = "/scratch/tmp/j_wred02/snni_campaign/moai-cpu"

DEF = f"""Bootstrap: localimage
From: {B}/sif/moai-cpu-{PARENT}.sif

# MOAI-CPU instrument image: `{PARENT}` PLUS the recording SEAL pool, and nothing else.
#
# WHY THE PARENT IS A STEP IMAGE. `MANIFEST.md` refuses the EARLIER CAMPAIGN's image as a baseline
# because it carries that campaign's instrumentation AND one of its levers. The parent here is this
# campaign's own published image for `{PARENT}`, so the binary below differs from the measured one
# by the recorder alone. That is exactly what has to be true for the peak comparison to mean
# anything: if the peak moves, only the recorder can have moved it.
#
# WHAT THE RECORDER IS. `MemoryPoolRecordingMT` keeps `MemoryPoolMT`'s policy unchanged -- same free
# lists, same retention, same growth -- and writes who takes and who returns an item into pmrec's
# own shared-mmap table format. `poll_peak.sh` samples it with `pmsample` inside the freeze at the
# peak, so the pool table and the host table describe ONE instant and both are in RESIDENT bytes.
# It is off unless SNNI_POOLREC=1, and then SNNI_POOLREC_TABLE names the mapping.
#
# Why this is needed at all: on this system a destroyed Ciphertext returns its block to SEAL's pool
# and never to `free`, so pmrec sees no release and its table names who made the POOL GROW rather
# than what is ALIVE at the peak.

%files
    {B}/patch_recording_pool.py /opt/patch_recording_pool.py

%post
    set -ex

    python3 /opt/patch_recording_pool.py /root/moai

    # SEAL first: the recorder lives in it, and the project links it with find_package, so a stale
    # install would be linked silently.
    cd /root/moai/thirdparty/SEAL-4.1-bs
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j "$(nproc)"
    cmake --install build

    cd /root/moai
    rm -rf build
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j "$(nproc)"
    ls -la /root/moai/build/test

    # WHAT THE PARENT PROMISED, RE-CHECKED ON THE NEW BINARY. A rebuild that lost any of these
    # would still run, still measure, and hand back a plausible number.
    strings -a /root/moai/build/test | grep -q 'SEEDED|seal_prng|' \\
        || {{ echo "FAILED: no SEAL seed marker in the binary"; exit 4; }}
    strings -a /root/moai/build/test | grep -q 'WORKLOAD|model=bert-base' \\
        || {{ echo "FAILED: no workload marker in the binary"; exit 4; }}
    strings -a /root/moai/build/test | grep -q 'MEM|' \\
        || {{ echo "FAILED: no MEM marker in the binary"; exit 4; }}
    strings -a /root/moai/build/test | grep -q 'SNNI_PARAM|' \\
        || {{ echo "FAILED: no parameter marker in the binary"; exit 4; }}
    readelf -S /root/moai/build/test | grep -q debug_info \\
        || {{ echo "FAILED: no debug_info; call sites would be inferred rather than measured"; exit 4; }}

    # THE RECORDER ITSELF. This image exists for one class.
    strings -a /root/moai/build/test | grep -q 'POOLREC|profile_installed' \\
        || {{ echo "FAILED: no POOLREC install marker in the binary"; exit 4; }}
    nm -C /root/moai/build/test | grep -q 'MemoryPoolRecordingMT' \\
        || {{ echo "FAILED: MemoryPoolRecordingMT is not linked into the binary"; exit 4; }}

    # NO LEVER MAY HAVE APPEARED that the parent did not have. This image adds an instrument, and
    # an instrument that also changed the program would answer a different question.
    strings -a /root/moai/build/test | grep -q 'SNNI_MM|profile=new' \\
        && {{ echo "FAILED: this image must NOT carry the pool policy change"; exit 4; }}
    strings -a /root/moai/build/test | grep -q 'LEVER|rtn_release_early' \\
        && {{ echo "FAILED: rtn_release_early rode along"; exit 4; }}
    strings -a /root/moai/build/test | grep -q 'LEVER|park_att_keys' \\
        && {{ echo "FAILED: park_att_keys rode along"; exit 4; }}
    strings -a /root/moai/build/test | grep -q 'LEVER|x_copy_release' \\
        && {{ echo "FAILED: x_copy_release rode along"; exit 4; }}

    nm -D /root/moai/build/test \\
        | grep -E ' U (malloc|calloc|realloc|aligned_alloc|posix_memalign|memalign|valloc|_Znwm|_Znam)(@.*)?$' \\
        | tee /opt/moai_alloc_imports.txt
    [ -s /opt/moai_alloc_imports.txt ] || {{ echo "FAILED: no allocator imports found at all"; exit 4; }}

    md5sum /root/moai/build/test > /opt/moai_binary_md5.txt
    cat /opt/moai_binary_md5.txt
    echo "{PARENT}" > /opt/poolrec_parent.txt

%environment
    export LC_ALL=C

%runscript
    echo "MOAI-CPU instrument image: {PARENT} + recording SEAL pool."
"""

io.open(OUT, "w", encoding="utf-8").write(DEF)
print("wrote " + OUT)
